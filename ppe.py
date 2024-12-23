from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import os

# Video Input
video_path = input("Enter the path to the video file: ")
if not os.path.exists(video_path):
    print("Error: Video file not found!")
    exit()

cap = cv2.VideoCapture(video_path)  # Open video file

# Load the YOLO model
model = YOLO("D:\\Infosys springboard\\PPE_kit_detection\\PPE_detection_Kit\\best.pt")

# Define class names
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

prev_frame_time = 0
fps = 0  # Initialize fps variable

# Check if video is opened
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Video Processing Loop
while True:
    success, img = cap.read()
    if not success:
        print("End of video or failed to capture frame.")
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 255, 0))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            label = f'{classNames[cls]} {conf:.2f}'
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate FPS
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time

    # Display FPS on the image
    cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)

    print(f'FPS: {fps}')

    cv2.imshow("PPE Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
