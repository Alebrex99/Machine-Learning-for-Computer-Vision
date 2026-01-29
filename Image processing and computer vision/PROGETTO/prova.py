import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    #print(f'{folderPath}/{imgPath}')
    overlayList.append(image)

print(len(overlayList))

while True:
    success, img = cap.read()



    cv2.imshow("Image", img)
    cv2.waitKey(1)



