# System Architecture

The system is designed as a modular, scalable architecture with clear separation of responsibilities.

## Core Components

### Camera Module
- Captures live video feed from CCTV/IP cameras
- Streams frames to the backend for processing

### Detection Engine
- YOLO model detects objects in each frame
- Outputs bounding boxes and confidence scores

### Tracking Module
- SORT tracks detected objects across frames
- Helps identify movement and persistence of objects

### Threat Analysis Engine
- Classifies detected objects into Safe, Suspicious, or Weapon
- Assigns threat levels based on predefined rules

### Alert System
- Triggers alerts for high and critical threats
- Sends notifications to the dashboard and authorities

### Dashboard
- Displays live detections
- Shows analytics and threat statistics
- Allows report export
