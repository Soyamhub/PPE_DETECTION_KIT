from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import os

choice = input("Enter '1' for Camera or '2' for Video: ")

if choice == '1':
    cap = cv2.VideoCapture(0)  # Open webcam
    cap.set(3, 640)  # Set frame width
    cap.set(4, 480)  # Set frame height
elif choice == '2':
    video_path = input("Enter the path to the video file: ")
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        exit()
    cap = cv2.VideoCapture(video_path)  # Open video file
else:
    print("Invalid choice! Exiting...")
    exit()

# Load the YOLO model
model = YOLO("D:\\Infosys springboard\\PPE_kit_detection\\PPE_detection_Kit\\best.pt")

# Define class names
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

prev_frame_time = 0
fps = 0  # Initialize fps variable

# Check if camera or video is opened
if not cap.isOpened():
    print("Error: Camera/Video not found or could not be opened.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image or end of video.")
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
