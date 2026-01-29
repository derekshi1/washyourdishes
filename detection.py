import cv2
import numpy as np
import time
from ultralytics import YOLO

# 1. CONFIGURATION
# ----------------
# COCO Class IDs: 41='cup', 45='bowl'. 
# Standard COCO does not have 'plate', so we monitor cups/bowls for the demo.
TARGET_CLASSES = [41, 45] 

# Alert threshold in seconds (Set to 600 for 10 minutes)
# We use 5 seconds here so you can test it quickly.
TIME_THRESHOLD = 5 

# 2. INITIALIZE MODEL & VIDEO
# ---------------------------
# Load a pre-trained YOLO model (n=nano is fastest)
model = YOLO('yolov8n.pt') 

# Open video file or webcam (change to 0 for webcam)
cap = cv2.VideoCapture(0) 

# Dictionary to store entry times: { track_id: timestamp }
dish_timers = {}

# 3. DEFINE SINK ZONE
# -------------------
# We'll define a polygon for the sink. 
# For this demo, we use a rectangle in the middle of a 640x480 frame.
# Format: np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], np.int32)
sink_polygon = np.array([
    [100, 100], 
    [540, 100], 
    [540, 380], 
    [100, 380]
], np.int32)

def is_point_in_sink(point, polygon):
    # Returns 1 if inside, -1 if outside, 0 if on edge
    return cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistent processing speed (optional)
    frame = cv2.resize(frame, (640, 480))
    
    # 4. RUN TRACKING
    # ---------------
    # persist=True is CRITICAL. It tells YOLO to keep ID numbers consistent 
    # between frames (Track ID 1 stays ID 1).
    results = model.track(frame, persist=True, classes=TARGET_CLASSES, verbose=False)

    # Visualize the sink zone
    cv2.polylines(frame, [sink_polygon], True, (0, 255, 0), 2)
    cv2.putText(frame, "SINK ZONE", (110, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 5. PROCESS DETECTIONS
    # ---------------------
    if results[0].boxes.id is not None:
        # Get boxes (xyxy), track IDs, and classes
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()

        current_time = time.time()
        
        # Identify which tracks are currently visible to handle "exit" logic if needed
        visible_ids = set()

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            visible_ids.add(track_id)
            
            # Calculate center point of the object
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Check if center is in sink
            if is_point_in_sink((center_x, center_y), sink_polygon):
                
                # If this is a new dish entering the sink, start timer
                if track_id not in dish_timers:
                    dish_timers[track_id] = current_time
                
                # Calculate duration
                duration = current_time - dish_timers[track_id]
                
                # Determine Color (Red if over time, Orange if active)
                color = (0, 165, 255) # Orange
                status_text = f"{int(duration)}s"
                
                if duration > TIME_THRESHOLD:
                    color = (0, 0, 255) # Red (Alert)
                    status_text = f"ALERT! {int(duration)}s"
                    # HERE IS WHERE YOU WOULD LOG THE TIMESTAMP TO A FILE
                    # print(f"Alert: Dish {track_id} in sink for > 10 mins")

                # Draw bounding box and timer
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID:{track_id} {status_text}", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            else:
                # If object is visible but NOT in sink, verify if we should remove timer
                # (Optional: Add logic here to remove ID from dish_timers if it leaves)
                if track_id in dish_timers:
                    del dish_timers[track_id]

    # Display the frame
    cv2.imshow("Sink Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()