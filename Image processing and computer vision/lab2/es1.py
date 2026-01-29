import cv2  # opencv
import numpy as np  # numpy
import matplotlib.pyplot as plt

#ES1-------------------------------------------------------------------------
location = "images/Poli.jpg"
img = cv2.imread(location)
plt.subplot(3, 2, 1).axis("off") #img mostrerà range di pixel nell'img
plt.imshow(img[0:200, 0:200])
plt.title("BGR Image")

#RGB cambio scala colore
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(3, 2, 2) #i primi 2 numeri indicano la dimensione
plt.imshow(imgRGB[0:200, 0:200])
plt.title("RGB Image");

#HSV
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.subplot(3, 2, 3) #i primi 2 numeri indicano la dimensione
plt.imshow(imgHSV[0:200, 0:200])
plt.title("HSV Image");

#punto 5 rimpiazzo parte img con matrice di 0
r = imgRGB[:,:,0] #matrice canale rosso
r[:,:] = 0 #azzero canale rosso
imgRGB[:,:,0] = r
plt.subplot(3,2,4)
plt.imshow(imgRGB)

#punto 6
r[:,:] = 255  #imgRGB[:,:,0]=255 #metto canale rosso a max intensità
imgRGB[:,:,0] = r
plt.subplot(3,2,5)
plt.imshow(imgRGB)


plt.show()



