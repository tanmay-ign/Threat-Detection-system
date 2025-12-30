import cv2
import time
import numpy as np
import os
import requests
from ultralytics import YOLO
from hand_gesture_detector import HandGestureDetector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000") + "/api/detect/upload"
print(f"[CONFIG] Backend URL: {BACKEND_URL}")

def send_to_backend(cropped_object, object_type, category, threat_level, camera_id, unique_object_id, confidence=None):
  
    try:
        if cropped_object is None or cropped_object.size == 0:
            print(f"[BACKEND ERROR] Cropped image is empty for {category}")
            return False
        
        crop_height, crop_width = cropped_object.shape[:2]
        if crop_height == 0 or crop_width == 0:
            print(f"[BACKEND ERROR] Invalid crop dimensions: {crop_width}x{crop_height}")
            return False
        
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        _, buffer = cv2.imencode('.jpg', cropped_object, encode_params)
        
        files = {
            'image': ('detection.jpg', buffer.tobytes(), 'image/jpeg')
        }
        
        data = {
            'object_type': object_type,
            'category': category,
            'threat_level': threat_level,
            'camera_id': camera_id,
            'unique_object_id': unique_object_id  # Send unique ID for deduplication
        }
        
        if confidence is not None:
            data['confidence'] = confidence
        
        response = requests.post(BACKEND_URL, files=files, data=data, timeout=5)
        
        if response.status_code == 200:
            print(f"[BACKEND] ✓ Uploaded {unique_object_id} ({threat_level}) [{crop_width}x{crop_height}]")
            return True
        elif response.status_code == 409:
            print(f"[BACKEND] ⊗ Duplicate {unique_object_id} (already in database)")
            return False
        else:
            print(f"[BACKEND ERROR] {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[BACKEND ERROR] Connection failed: {e}")
        return False
    except Exception as e:
        print(f"[BACKEND ERROR] {e}")
        return False

CONF = 0.45
IOU = 0.5
IMG_SIZE = 640

FRAME_SKIP = 2
PROXIMITY_THRESHOLD = 150

LOW_TIME = 10
MED_TIME = 25
HIGH_TIME = 45
CRITICAL_TIME = 60  # Bags unattended for > 60s are CRITICAL

SNAPSHOT_DIR = "saved_frames"
WEAPON_DIR = os.path.join(SNAPSHOT_DIR, "weapons")
BAG_DIR = os.path.join(SNAPSHOT_DIR, "bags")
NORMAL_DIR = os.path.join(SNAPSHOT_DIR, "normal_objects")
PERSON_DIR = os.path.join(SNAPSHOT_DIR, "persons")

os.makedirs(WEAPON_DIR, exist_ok=True)
os.makedirs(BAG_DIR, exist_ok=True)
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(PERSON_DIR, exist_ok=True)

COLORS = {
    "person": (0, 165, 255),
    "bag": (255, 0, 255),
    "weapon": (0, 0, 255),
    "benign": (0, 255, 0),  
    "attended": (0, 255, 0),
    "medium": (0, 255, 255),
    "high": (0, 0, 255)
}

BAG_CLASSES = ["backpack", "handbag", "suitcase"]
WEAPON_CLASSES = ["knife", "scissors", "gun", "pistol", "rifle"]
SAFE_OBJECTS = ["bottle", "book", "cup", "laptop", "phone", "bowl", "chair", "dining table", 
                "sink", "oven", "refrigerator", "tv", "remote", "keyboard", "mouse", "cell phone", "key"]

def determine_threat_level(object_name, object_type=None, time_unattended=0):

    if object_name in WEAPON_CLASSES:
        return "CRITICAL"
    
    if object_name in BAG_CLASSES:
        if time_unattended > CRITICAL_TIME:
            return "CRITICAL"  
        elif time_unattended > HIGH_TIME:
            return "HIGH"  
        elif time_unattended > MED_TIME:
            return "MEDIUM"  
        else:
            return "SAFE" 
    
    if object_type == "person":
        return "SAFE"
    
    return "SAFE"

def crop_with_padding(frame, x1, y1, x2, y2, padding_percent=0.20):
   
    try:
        h, w = frame.shape[:2]
        
        box_width = x2 - x1
        box_height = y2 - y1
        
        pad_x = int(box_width * padding_percent)
        pad_y = int(box_height * padding_percent)
        
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w, x2 + pad_x)
        y2_pad = min(h, y2 + pad_y)
        
        cropped = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
            print(f"[CROP ERROR] Invalid crop dimensions: {cropped.shape}")
            return None
        
        crop_h, crop_w = cropped.shape[:2]
        max_dimension = 320
        
        if crop_h > max_dimension or crop_w > max_dimension:
            scale = max_dimension / max(crop_h, crop_w)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            
            cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return cropped
        
    except Exception as e:
        print(f"[CROP ERROR] {e}")
        return None

model = YOLO("yolov8m.pt")

hand_detector = HandGestureDetector()
gesture_mode = False  

MOBILE_IP = "http://192.168.29.177:8080"   

cap = None
camera_source = "Webcam"

if MOBILE_IP.strip():
    print("[INFO] Trying mobile camera...")
    cap = cv2.VideoCapture(MOBILE_IP + "/video", cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)  
    time.sleep(2) 

    ret, test_frame = cap.read()
    if ret and test_frame is not None:
        camera_source = "Mobile"
        print("[OK] Mobile camera connected")
    else:
        print("[WARN] Mobile camera failed, switching to webcam")
        cap.release()
        cap = None

if cap is None:
    cap = cv2.VideoCapture(0)
    camera_source = "Webcam"
    print("[OK] Webcam connected")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)  

if not cap.isOpened():
    print("Camera error")
    exit()

object_state = {} 
prev_time = time.time()
frame_id = 0

print("SMART THREAT DETECTION SYSTEM RUNNING")
print("Camera Source:", camera_source)
print("Press 'q' to quit")
print("Press 'h' to toggle Hand Gesture Mode")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[WARN] Frame not received, retrying...")
        if camera_source == "Mobile":
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(MOBILE_IP + "/video", cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            continue
        else:
            break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue

    if frame.size == 0:
        continue
        
    frame = cv2.resize(frame, (640, 480))
    now = time.time()
    
    clean_frame = frame.copy()

    if gesture_mode:
        frame, gestures = hand_detector.process_frame(frame)
        frame = hand_detector.draw_gestures(frame, gestures)
        
        cv2.putText(frame, "MODE: HAND GESTURE DETECTION",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Hands Detected: {len(gestures)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, "Press 'h' to switch to Threat Detection",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    else:
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF,
            iou=IOU,
            imgsz=IMG_SIZE,
            verbose=False
        )

        persons = []
        objects = []

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                name = model.names[cls]

                unique_object_id = f"{camera_source}_{name}_{tid}"
                
                if unique_object_id not in object_state:
                    object_state[unique_object_id] = {
                        "first_seen_time": now,
                        "last_seen": now,
                        "alert_sent": False,
                        "image_saved": False
                    }
                    print(f"[NEW OBJECT] Detected: {unique_object_id}")
                else:
                    object_state[unique_object_id]["last_seen"] = now

                objects.append((unique_object_id, name, (x1,y1,x2,y2), (cx,cy)))

                if name == "person":
                    persons.append((cx,cy))

        threat_level = "SAFE"

        for unique_object_id, name, box, center in objects:
            x1,y1,x2,y2 = box
            cx,cy = center

            obj_state = object_state[unique_object_id]

            if name == "person":
                cv2.rectangle(frame,(x1,y1),(x2,y2),COLORS["person"],2)
                cv2.putText(frame,"Person",(x1,y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS["person"],2)
                
                if obj_state["image_saved"]:
                    continue
                
                person_threat = determine_threat_level(name, object_type="person")
                cropped_object = crop_with_padding(clean_frame, x1, y1, x2, y2, padding_percent=0.20)
                if cropped_object is not None and cropped_object.size > 0:
                    success = send_to_backend(cropped_object, "person", "person", person_threat, camera_source, unique_object_id)
                    obj_state["image_saved"] = True
                    if success:
                        print(f"[SAVED] {unique_object_id}")
                    else:
                        print(f"[SKIP] {unique_object_id} - duplicate in database")
                continue

            if name in WEAPON_CLASSES:
                weapon_threat = determine_threat_level(name, object_type="weapon")
                threat_level = weapon_threat  # Update frame-level threat
                
                cv2.rectangle(frame,(x1,y1),(x2,y2),COLORS["weapon"],3)
                cv2.putText(frame,f"WEAPON: {name.upper()}",
                            (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,COLORS["weapon"],2)

                if obj_state["alert_sent"] and obj_state["image_saved"]:
                    continue
                
                cropped_object = crop_with_padding(clean_frame, x1, y1, x2, y2, padding_percent=0.20)
                if cropped_object is not None and cropped_object.size > 0:
                    success = send_to_backend(cropped_object, "weapon", name, weapon_threat, camera_source, unique_object_id)
                    obj_state["alert_sent"] = True
                    obj_state["image_saved"] = True
                    if success:
                        print(f"[ALERT SENT] {unique_object_id}")
                    else:
                        print(f"[SKIP] {unique_object_id} - duplicate in database")
                continue

            if name in BAG_CLASSES:
                attended = False
                for px,py in persons:
                    if np.hypot(cx-px, cy-py) < PROXIMITY_THRESHOLD:
                        attended = True
                        break

                t = now - obj_state["first_seen_time"]

                bag_threat = determine_threat_level(name, object_type="bag", time_unattended=t)

                if attended:
                    color = COLORS["attended"]
                    label = f"{name} | ATTENDED"
                elif t > CRITICAL_TIME:
                    color = COLORS["high"]  # Red for critical
                    label = f"{name} | CRITICAL {int(t)}s"
                    threat_level = "CRITICAL"
                elif t > HIGH_TIME:
                    color = COLORS["high"]
                    label = f"{name} | UNATTENDED {int(t)}s"
                    threat_level = "HIGH"
                elif t > MED_TIME:
                    color = COLORS["medium"]
                    label = f"{name} | SUSPICIOUS {int(t)}s"
                else:
                    color = COLORS["bag"]
                    label = f"{name}"

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
                cv2.putText(frame,label,(x1,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                
                if bag_threat == "CRITICAL":
                    if obj_state["image_saved"]:
                        continue
                    
                    print(f"[CRITICAL BAG] {unique_object_id} unattended for {int(t)}s")
                    cropped_object = crop_with_padding(clean_frame, x1, y1, x2, y2, padding_percent=0.20)
                    if cropped_object is not None and cropped_object.size > 0:
                        success = send_to_backend(cropped_object, "bag", name, bag_threat, camera_source, unique_object_id)
                        obj_state["image_saved"] = True
                        if success:
                            print(f"[SAVED] {unique_object_id} as CRITICAL")
                    else:
                        print(f"[SKIP] {unique_object_id} - duplicate in database")
                continue

            cv2.rectangle(frame,(x1,y1),(x2,y2),COLORS["benign"],2)
            cv2.putText(frame,name,(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS["benign"],2)
            
            if obj_state["image_saved"]:
                continue
            
            object_threat = determine_threat_level(name, object_type="object")
            cropped_object = crop_with_padding(clean_frame, x1, y1, x2, y2, padding_percent=0.20)
            if cropped_object is not None and cropped_object.size > 0:
                success = send_to_backend(cropped_object, "object", name, object_threat, camera_source, unique_object_id)
                obj_state["image_saved"] = True
                if success:
                    print(f"[SAVED] {unique_object_id}")
                else:
                    print(f"[SKIP] {unique_object_id} - duplicate in database")

        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now

        cv2.putText(frame,f"THREAT: {threat_level}",
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,
                    (0,0,255) if threat_level!="SAFE" else (0,255,0),2)

        cv2.putText(frame,f"FPS: {int(fps)} | Source: {camera_source}",
                    (10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        
        cv2.putText(frame, "Press 'h' for Hand Gesture Mode",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("SMART THREAT DETECTION", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        gesture_mode = not gesture_mode
        mode_name = "Hand Gesture" if gesture_mode else "Threat Detection"
        print(f"[MODE] Switched to {mode_name} Mode")

cap.release()
hand_detector.release()
cv2.destroyAllWindows()
print("System stopped.")
print(f"Persons saved to: {PERSON_DIR}")
print(f"Weapons saved to: {WEAPON_DIR}")
print(f"Bags saved to: {BAG_DIR}")
print(f"Normal objects saved to: {NORMAL_DIR}")