"""
https://www.datacamp.com/tutorial/face-detection-python-opencv
"""
import cv2
import matplotlib.pyplot as plt

imagePath = 'sources/test.png'

# Read the image
img = cv2.imread(imagePath)
if img is None:
    print(f"Error: Unable to load image at {imagePath}")
else:
    print(img.shape)

# Convert to grayscale
if img is not None:
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray_image.shape)

    # Load the face classifier
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_classifier.empty():
        print("Error: Failed to load face classifier")
    else:
        # Detect faces
        face = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        # Draw rectangles around faces
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # Convert to RGB and display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()

