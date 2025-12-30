# Backend Implementation

The backend is built using **Python and FastAPI**, chosen for their performance and scalability.

## Key Responsibilities
- Access live camera feeds using OpenCV
- Process video frames in real time
- Run YOLO inference for object detection
- Track objects using SORT
- Classify threat levels
- Store detection data in MongoDB
- Expose APIs for frontend consumption

## Database Design
MongoDB stores detection records under categories:
- Safe objects
- Suspicious objects
- Weapon detections

Each record includes timestamp, object type, threat level, confidence score, and camera source.
