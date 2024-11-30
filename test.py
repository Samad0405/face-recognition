from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime
import csv

# Initialize webcam and face detection model
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')  # Ensure path is correct

# Load face data and labels (names)
try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
except FileNotFoundError:
    print("Error: Required files not found in 'data/'. Ensure 'names.pkl' and 'faces_data.pkl' exist.")
    exit()

# Train the KNN classifier with face data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Set up attendance file for today
today_date = datetime.now().strftime("%d-%m-%Y")
attendance_file = f"data/Attendance_{today_date}.csv"
COL_NAMES = ['NAME', 'TIME']

# Load existing attendance records (if any) to avoid duplicates
logged_names = set()
if os.path.isfile(attendance_file):
    with open(attendance_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            if row:
                logged_names.add(row[0])  # Add the name to the set

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)[0]

        # Only log attendance if the person is not already in today's log
        if output not in logged_names:
            # Display name on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(frame, output, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            # Capture timestamp and log attendance
            timestamp = datetime.now().strftime("%H:%M:%S")
            attendance = [output, timestamp]
            
            # Write to CSV
            with open(attendance_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if not logged_names:  # Write header if the file was empty
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            
            # Add name to logged set to avoid re-logging
            logged_names.add(output)

    cv2.imshow("Frame", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
