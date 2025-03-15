from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import pyttsx3

faceCascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
vidCapt = cv2.VideoCapture(0)

imgBackground = cv2.imread("image.png")

with open ('data/names.pkl', 'rb') as f:
    LABELS =  pickle.load(f)
with open ('data/facesData.pkl', 'rb') as f:
    FACES =  pickle.load(f)

object = KNeighborsClassifier(n_neighbors=5)
object.fit(FACES, LABELS)

NamesCol = ['Name of Student', 'Time when attendance marked']

while True:
    ret,frame = vidCapt.read()
    colTrans = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = faceCascade.detectMultiScale(colTrans, 1.3, 6)

    for (x1, y1, w1, h1) in f:
        cropImage = frame[y1:y1+h1, x1:x1+w1, :]
        resizedImage = cv2.resize(cropImage, (50,50)).flatten().reshape(1, -1)
        output = object.predict(resizedImage)
        t = time.time()
        date = datetime.fromtimestamp(t).strftime("%d-%m-%Y")
        timeStamp = datetime.fromtimestamp(t).strftime("%H-%M-%S")
        

        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1 - 40), (x1 + w1, y1), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, str(output[0]), (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.82, (255, 255, 255), 1)
        attendance = [str(output[0]), str(timeStamp)]


    imgBackground[100:100+480, 110:110+640] = frame
    
    cv2.imshow("WebCam", imgBackground)
    waitTime = cv2.waitKey(1) & 0xff
    if waitTime == ord('w'):
        file_path = "Attendance/Attendance_" + date + ".csv"
        existingFile = os.path.isfile(file_path)

        mode = "a" if existingFile else "w"

        with open(file_path, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)

            if not existingFile:
                writer.writerow(NamesCol)

            writer.writerow(attendance)
            sound = pyttsx3.init()
            sound.say("Attendance Marked")
            sound.runAndWait()

    if waitTime == ord('q'):
        break

vidCapt.release()
cv2.destroyAllWindows()
