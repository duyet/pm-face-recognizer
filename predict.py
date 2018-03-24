
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("model/trainingdata.yml")
id = 0

font = cv2.FONT_HERSHEY_DUPLEX

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        if id == 1 or id == 10:
            id="Duyet"
        if id ==2:
            id="Hung"

        cv2.putText(img, str(id), (x, y + h + 20), font, 0.8, (255, 255, 255), 1)
    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
