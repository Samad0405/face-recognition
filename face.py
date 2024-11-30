import cv2
import pickle
import numpy as np
import os

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Initialize webcam and face detection model
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faces_data = []  # List to store face data
i = 0
name = input("Enter Your Name: ")

# Capture face images
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize to 50x50 pixels
        if len(faces_data) <= 100 and i % 10 == 0:  # Capture every 10th frame
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        
    cv2.imshow("Frame", frame)
    
    # Exit on 'q' or when 100 face samples are captured
    if cv2.waitKey(1) == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# Convert face data to numpy array and reshape for storage
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

# Save or append face data
if os.path.exists('data/faces_data.pkl'):
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
else:
    faces = faces_data

with open('data/faces_data.pkl', 'wb') as f:
    pickle.dump(faces, f)

# Save or append names
name_entries = [name] * 100  # Store the name 100 times (one for each face)
if os.path.exists('data/names.pkl'):
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + name_entries
else:
    names = name_entries

with open('data/names.pkl', 'wb') as f:
    pickle.dump(names, f)
