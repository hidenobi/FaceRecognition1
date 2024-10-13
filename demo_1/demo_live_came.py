from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics import YOLO
import cv2
import numpy as np

# Load a pre-trained YOLOv10n model
model = YOLO("yolov10n.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Loop through frames from the webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    try:
        # Create a copy of the original frame
        original_frame = frame.copy()

        # Perform inference
        results = model(frame)

        # Render the results on the frame
        annotated_frame = results[0].plot()

        # Convert the annotated frame to a format compatible with cv2.imshow
        # annotated_frame = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

        # Display the original frame
        cv2.imshow('Original Frame', original_frame)

        # Display the frame with the bounding boxes
        cv2.imshow('YOLOv10 Live Detection', annotated_frame)

    except Exception as e:
        print(f"An error occurred: {e}")
        break

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()