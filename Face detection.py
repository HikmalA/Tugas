# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:09:31 2024

@author: HIKMAL AKBAR
"""

#face detection
import cv2 # menyertakan library cv2 dari opencv
import numpy as np

face_cascade = cv2.CascadeClassifier('C:/Users/Lenovo/Downloads/haarcascade_frontalface_default.xml')

image = cv2.imread("C:/Users/Lenovo/Downloads/WhatsApp Image 2024-11-20 at 10.22.16_d3a833f9.jpg")
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(img, scaleFactor=1.1)
for (x,y,w,h) in faces: #looping sejumlah wajah yang ditemukan (xy kordinat wh lebar tinggi)
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,155), 3)


cv2.imshow("display", image)
#cv2.imwrite("display", image)
cv2.waitKey(0)
cv2.destroyAllWindows()