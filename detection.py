import cv2
import numpy as np
import time
from ultralytics import YOLO

# --- CONFIGURATION ---
TARGET_CLASSES = [41, 45]  # COCO IDs: Cup, Bowl
ALERT_THRESHOLD = 5        # Seconds before alert (Set to 600 for 10 mins)

# --- STATE MANAGEMENT ---
# Dictionary structure: { track_id: { 'entry_time': float, 'status': 'VISIBLE'/'COVERED' } }
sink_inventory = {}

def get_sink_polygon():
    # Define a polygon for the sink (adjust these points for your video)
    # This example creates a box in the middle of a standard webcam feed
    return np.array([
        [150, 100], 
        [500, 100], 
        [500, 380], 
        [150, 380]
    ], np.int32)

def is_point_in_sink(point, polygon):
    # Returns True if inside, False if outside
    return cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0

def run():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0) # 0 for webcam, or path to video file

    sink_polygon = get_sink_polygon()
    
    print("Strict Monitor Started. Objects are only removed if seen EXITING the zone.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (800, 600))
        current_time = time.time()

        # 1. RUN TRACKING
        results = model.track(frame, persist=True, classes=TARGET_CLASSES, verbose=False, tracker="botsort.yaml")

        # Get all currently detected IDs in this specific frame
        current_frame_ids = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                current_frame_ids.add(track_id)
                
                x1, y1, x2, y2 = box
                center_point = (int((x1+x2)/2), int((y1+y2)/2))
                
                # Check where the object is RIGHT NOW
                is_inside = is_point_in_sink(center_point, sink_polygon)

                # --- LOGIC GATE ---
                
                # CASE A: Object is detected INSIDE the sink
                if is_inside:
                    # Draw Red Box (In Sink)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    
                    if track_id not in sink_inventory:
                        sink_inventory[track_id] = {
                            'entry_time': current_time,
                            'status': 'VISIBLE'
                        }
                    else:
                        sink_inventory[track_id]['status'] = 'VISIBLE'

                # CASE B: Object is detected OUTSIDE the sink (The "Exit" event)
                else:
                    # Draw Green Box (Safe Zone)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # If this object WAS in our inventory, it has now officially left.
                    if track_id in sink_inventory:
                        print(f"Object {track_id} removed (seen exiting zone).")
                        del sink_inventory[track_id]

        # 2. HANDLE HIDDEN OBJECTS
        # Check all items currently in our inventory.
        # If they are NOT in 'current_frame_ids', it means the camera lost them, mark as covered
        for trk_id in sink_inventory:
            if trk_id not in current_frame_ids:
                sink_inventory[trk_id]['status'] = 'COVERED'

        # 3. DRAW DASHBOARD
        cv2.polylines(frame, [sink_polygon], True, (255, 255, 0), 2)
        cv2.putText(frame, "SINK ZONE", (160, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        y_offset = 50
        cv2.putText(frame, "CURRENTLY IN SINK:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if not sink_inventory:
             cv2.putText(frame, "Empty", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        for trk_id, data in sink_inventory.items():
            duration = current_time - data['entry_time']
            status = data['status']
            
            # Formatting
            if status == 'COVERED':
                display_text = f"ID {trk_id} (HIDDEN): {int(duration)}s"
                color = (150, 150, 150) # Grey
            else:
                display_text = f"ID {trk_id} (SEEN): {int(duration)}s"
                color = (0, 255, 0) # Green

            if duration > ALERT_THRESHOLD:
                color = (0, 0, 255) # Red for Alert
                display_text += " [ALERT!]"

            cv2.putText(frame, display_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30

        cv2.imshow("Strict Sink Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()