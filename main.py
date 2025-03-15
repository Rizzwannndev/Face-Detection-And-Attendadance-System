import cv2
import pickle
import numpy as np
import os

faceCascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
vidCapt = cv2.VideoCapture(0)

facesData = []
name = input("Enter Your Name: ")

i = 0
while True:
    # Where ret refers to rectangle and frame refers to the image.
    ret,frame = vidCapt.read()
    colTrans = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = faceCascade.detectMultiScale(colTrans, 1.3, 6)

    for (x1, y1, w1, h1) in f:
        cropImage = frame[y1:y1+h1, x1:x1+w1, :]
        resizedImage = cv2.resize(cropImage, (50,50))
        if len(facesData)<=100 and i%10 == 0:
            facesData.append(resizedImage)
        i = i+1
        cv2.putText(frame, str(len(facesData)),(50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (50,50,255), 1)
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (88, 255, 50), 1)

    cv2.imshow("WebCam", frame)
    waitTime = cv2.waitKey(1) & 0xff
    if waitTime == ord('q') or len(facesData) == 100:
        break

vidCapt.release()
cv2.destroyAllWindows()

facesData = np.asarray(facesData)
facesData = facesData.reshape(100, -1)

# For Names File
if 'names.pkl' not in os.listdir('data/'):
    names = [name]*100
    with open ('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open ('data/names.pkl', 'rb') as f:
        names =  pickle.load(f)
    names = names + [name]*100
    with open ('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
# For Faces File
if 'facesData.pkl' not in os.listdir('data/'):
    with open ('data/facesData.pkl', 'wb') as f:
        pickle.dump(facesData, f)
else:
    with open ('data/facesData.pkl', 'rb') as f:
        faces =  pickle.load(f)
    faces = np.append(faces, facesData, axis=0)
    with open ('data/facesData.pkl', 'wb') as f:
        pickle.dump(faces, f)