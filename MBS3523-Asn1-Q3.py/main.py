import cv2
import time
import numpy as np
import imutils

cap = cv2.VideoCapture('rec/23.mp4')

car_cascade = cv2.CascadeClassifier('rec/cars.xml')
face=cv2.CascadeClassifier('rec/fullbody.xml')

while True:
    ret, frames = cap.read()
    gray=cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars= car_cascade.detectMultiScale(gray, 1.4, 2)
    hum= face.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in hum:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow('Detection', frames)
    if cv2.waitKey(100) == 27:
        break
cap.release()
cv2.destroyAllWindows()