import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model (using a larger model for better accuracy)
model = YOLO('yolov8m.pt')

# Open the video file
video_path = "../Devasandra_Sgnl_JN_FIX_3_time_2024-05-27T07-30-0.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the 5% and 10% lines
line_5_percent = 0.05
line_10_percent = 0.10

# Initialize lists to store entry and exit points
entry_points = []
exit_points = []

# Dictionary to store vehicle trajectories
vehicle_trajectories = {}

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Function to check if a point is between two lines
def is_between_lines(point, line1, line2):
    return min(line1, line2) <= point <= max(line1, line2)

# Function to check for entry or exit
def check_entry_exit(prev_pos, curr_pos):
    prev_x, prev_y = prev_pos
    curr_x, curr_y = curr_pos

    if is_between_lines(prev_x, 0, width * line_5_percent) and \
            is_between_lines(curr_x, width * line_5_percent, width * line_10_percent):
        return "entry", (curr_x, curr_y)

    elif is_between_lines(prev_x, width * (1 - line_5_percent), width) and \
            is_between_lines(curr_x, width * (1 - line_10_percent), width * (1 - line_5_percent)):
        return "entry", (curr_x, curr_y)

    elif is_between_lines(prev_y, 0, height * line_5_percent) and \
            is_between_lines(curr_y, height * line_5_percent, height * line_10_percent):
        return "entry", (curr_x, curr_y)

    elif is_between_lines(prev_y, height * (1 - line_5_percent), height) and \
            is_between_lines(curr_y, height * (1 - line_10_percent), height * (1 - line_5_percent)):
        return "entry", (curr_x, curr_y)

    elif (is_between_lines(prev_x, width * line_10_percent, width * line_5_percent) and \
          is_between_lines(curr_x, 0, width * line_5_percent)) or \
            (is_between_lines(prev_x, width * (1 - line_10_percent), width * (1 - line_5_percent)) and \
             is_between_lines(curr_x, width * (1 - line_5_percent), width)) or \
            (is_between_lines(prev_y, height * line_10_percent, height * line_5_percent) and \
             is_between_lines(curr_y, 0, height * line_5_percent)) or \
            (is_between_lines(prev_y, height * (1 - line_10_percent), height * (1 - line_5_percent)) and \
             is_between_lines(curr_y, height * (1 - line_5_percent), height)):
        return "exit", (curr_x, curr_y)

    return None, None

# Process video frames
frame_count = 0
previous_detections = []
next_vehicle_id = 0
last_valid_frame = None

while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            break

        last_valid_frame = frame.copy()
        frame_count += 1

        # Run YOLOv8 inference on the frame
        results = model(frame)

        current_detections = []

        # Process each detected object
        for det in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 2 and conf > 0.6:  # Higher confidence threshold for better accuracy
                current_detections.append((x1, y1, x2, y2))

        matched_indices = set()
        for i, curr_box in enumerate(current_detections):
            best_iou = 0
            best_match = -1
            for j, prev_box in enumerate(previous_detections):
                iou = calculate_iou(curr_box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = j

            if best_iou > 0.4:  # Adjusted IoU threshold
                matched_indices.add(best_match)
                vehicle_id = f"vehicle_{best_match}"
            else:
                vehicle_id = f"vehicle_{next_vehicle_id}"
                next_vehicle_id += 1

            if vehicle_id not in vehicle_trajectories:
                vehicle_trajectories[vehicle_id] = []

            center_x = (curr_box[0] + curr_box[2]) / 2
            center_y = (curr_box[1] + curr_box[3]) / 2
            vehicle_trajectories[vehicle_id].append((center_x, center_y))

            if len(vehicle_trajectories[vehicle_id]) >= 2:
                prev_pos = vehicle_trajectories[vehicle_id][-2]
                curr_pos = vehicle_trajectories[vehicle_id][-1]

                point_type, point = check_entry_exit(prev_pos, curr_pos)
                if point_type == "entry":
                    entry_points.append(point)
                    print(f"Entry point detected at frame {frame_count}: {point}")
                elif point_type == "exit":
                    exit_points.append(point)
                    print(f"Exit point detected at frame {frame_count}: {point}")

        previous_detections = current_detections

        if frame_count % 100 == 0:
            print(f"Processed frame {frame_count}/{total_frames} ({frame_count / total_frames * 100:.2f}%)")

    except Exception as e:
        print(f"Error processing frame {frame_count}: {str(e)}")
        continue

# Draw entry and exit points on the final frame
for point in entry_points:
    cv2.circle(last_valid_frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
for point in exit_points:
    cv2.circle(last_valid_frame, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

# Draw the 5% and 10% lines on the final frame
cv2.line(last_valid_frame, (int(width * line_5_percent), 0), (int(width * line_5_percent), height), (255, 255, 0), 1)
cv2.line(last_valid_frame, (int(width * line_10_percent), 0), (int(width * line_10_percent), height), (255, 255, 0), 1)
cv2.line(last_valid_frame, (int(width * (1 - line_5_percent)), 0), (int(width * (1 - line_5_percent)), height),
         (255, 255, 0), 1)
cv2.line(last_valid_frame, (int(width * (1 - line_10_percent)), 0), (int(width * (1 - line_10_percent)), height),
         (255, 255, 0), 1)
cv2.line(last_valid_frame, (0, int(height * line_5_percent)), (width, int(height * line_5_percent)), (255, 255, 0), 1)
cv2.line(last_valid_frame, (0, int(height * line_10_percent)), (width, int(height * line_10_percent)), (255, 255, 0), 1)
cv2.line(last_valid_frame, (0, int(height * (1 - line_5_percent))), (width, int(height * (1 - line_5_percent))),
         (255, 255, 0), 1)
cv2.line(last_valid_frame, (0, int(height * (1 - line_10_percent))), (width, int(height * (1 - line_10_percent))),
         (255, 255, 0), 1)

# Save the final frame with entry and exit points
if last_valid_frame is not None:
    try:
        cv2.imwrite('../final_frame.jpg', last_valid_frame)
        print("Final frame saved successfully.")
    except Exception as e:
        print(f"Error saving final frame: {str(e)}")
else:
    print("No valid frames were processed. Unable to save final frame.")

print(f"Total entry points: {len(entry_points)}")
print(f"Total exit points: {len(exit_points)}")
print(f"Total unique vehicles detected: {len(vehicle_trajectories)}")
