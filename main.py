import cv2
from ultralytics import YOLO
import numpy as np
import time

# Loading a YOLO model
model = YOLO("yolo11s.pt")

# Open the input video
input_video_path = "5.mp4"
output_video_path = "output_video2.mp4"
cap = cv2.VideoCapture(input_video_path)

# Getting video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Keeping the original FPS but update every 60 frames
frame_counter = 0
update_interval = 20

# Defining the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Defining a threshold for traffic density
traffic_density_threshold = 4  # Adjust this value based on your needs

# Initializing variables for polygon drawing
polygon_points = []
drawing_polygon = False
vehicle_count = 0  # Store vehicle count for every update interval

# Variable to store the signal duration in seconds
signal_duration = 15  # Set the desired duration in seconds

# Variable to track the start time of the signal
signal_start_time = None

# Variable to track the signal state
signal_state = "OFF"

# Mouse callback function to draw polygon
def draw_polygon(event, x, y, flags, param):
    global drawing_polygon, polygon_points, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_polygon = True
        polygon_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    elif event == cv2.EVENT_MOUSEMOVE and drawing_polygon:
        if len(polygon_points) > 1:
            temp_frame = frame.copy()
            cv2.line(temp_frame, polygon_points[-1], (x, y), (0, 255, 0), 2)
            cv2.imshow('Draw Polygon', temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_polygon = False
        polygon_points.append((x, y))
        if len(polygon_points) > 1:
            cv2.line(frame, polygon_points[-2], polygon_points[-1], (0, 255, 0), 2)

# Reading the first frame to draw the polygon
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    exit()

# Creating a window and bind the mouse callback function
cv2.namedWindow('Draw Polygon')
cv2.setMouseCallback('Draw Polygon', draw_polygon)

# Drawing the polygon on the first frame
while True:
    cv2.imshow('Draw Polygon', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: 
        break
    elif k == 13: 
        break

cv2.destroyAllWindows()

# Checking if the polygon was drawn
if len(polygon_points) < 3:
    print("Error: Polygon must have at least 3 points.")
    cap.release()
    out.release()
    exit()

# Converting the list of points to a NumPy array and ensure they are integers
polygon_points = np.array(polygon_points, dtype=np.int32)
polygon_points = polygon_points.reshape((-1, 1, 2))

# Processing each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    
    # Only update vehicle count every 60 frames
    if frame_counter % update_interval == 0:
        # Converting frame to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Performing prediction
        results = model.predict(frame_rgb, imgsz=640, classes=[1, 2, 3, 5, 7])

        vehicle_count = 0  # Reset vehicle count for the update interval

        # Counting vehicles inside the polygon
        if isinstance(results, list):
            for result in results:
                for det in result.boxes:
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    
                    # Checking if the center of the detected object is inside the polygon
                    box_center = (int((x1 + x2) // 2), int((y1 + y2) // 2))
                    if cv2.pointPolygonTest(polygon_points, box_center, False) >= 0:
                        vehicle_count += 1

    # Checking if the signal should be turned ON based on vehicle count
    if vehicle_count > traffic_density_threshold and signal_state == "OFF":
        signal_start_time = time.time() 
        signal_state = "ON"

    # Checking if the signal duration has passed
    if signal_start_time is not None and time.time() - signal_start_time >= signal_duration:
        signal_state = "OFF"
        signal_start_time = None 

    # Determining the color and message based on the signal state
    if signal_state == "ON":
        color = (0, 255, 0)  # Green for high traffic
        message = "Signal Turn ON"
    else:
        color = (0, 0, 255)  # Red for low traffic
        message = "Signal Turn OFF"

    # Creating a filled polygon with transparency
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_points], color)
    
    # Blending the overlay with the original frame
    alpha = 0.4  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Drawing polygon outline
    cv2.polylines(frame, [polygon_points], isClosed=True, color=(255, 255, 255), thickness=2)

    # Displaying the signal status in an attractive manner
    cv2.putText(frame, message, (frame_width // 2 - 150, frame_height // 5),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5, cv2.LINE_AA)

    # Writing the frame to the output video
    out.write(frame)

# Releas everything
cap.release()
out.release()
cv2.destroyAllWindows()