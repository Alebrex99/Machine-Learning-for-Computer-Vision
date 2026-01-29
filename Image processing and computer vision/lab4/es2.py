import cv2
import matplotlib.pyplot as plt
import numpy as np

location = "images/snowy-street.jpg"
img = cv2.imread(location)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(3, 2, 1).axis("off")
plt.imshow(imgRGB)
plt.title("RGB Image")

imgGRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#CLOSING: DILATO + ERODO
kernel = np.ones((2,2),np.uint8)
dilation = cv2.dilate(imgRGB,kernel,iterations = 3)
plt.subplot(3, 2, 2).axis("off")
plt.imshow(dilation)
plt.title("DILATION Image- chiusura")

erosion = cv2.erode(imgRGB,kernel,iterations = 3)
plt.subplot(3, 2, 3).axis("off")
plt.imshow(erosion)
plt.title("EROSION Image - chiusura")


#ES2-------------------------------------------------------
#OPENING: ERODO + DILATO
erosion2 = cv2.erode(imgRGB,kernel,iterations = 3)
plt.subplot(3, 2, 4).axis("off")
plt.imshow(erosion2)
plt.title("EROSION2 Image")

dilation2 = cv2.dilate(imgRGB,kernel,iterations = 3)
plt.subplot(3, 2, 5).axis("off")
plt.imshow(dilation2)
plt.title("DILATION2 Image")

plt.show()