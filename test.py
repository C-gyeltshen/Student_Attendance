import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from tensorflow.keras.models import load_model

# Load the CNN model
cnn_model = load_model("path/to/your/saved/model.h5")

def preprocess_image(img):
    return cv2.resize(img, (50, 50))  # Resize to match input size of CNN

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load names and faces data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

# Define column names for the attendance CSV file
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = preprocess_image(crop_img) 

        prediction = cnn_model.predict(np.expand_dims(resized_img, axis=0))

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(prediction[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        attendance = [str(prediction[0]), str(timestamp)]

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        
        if k == ord('o'):
            if exist:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
            else:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)
        elif k == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
