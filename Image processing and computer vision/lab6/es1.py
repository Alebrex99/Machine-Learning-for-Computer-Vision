import cv2  # opencv
import numpy as np  # numpy
import matplotlib.pyplot as plt

"""prendiamo HSV perchè una volta scelto H siamo sicuri di aver preso quel tipo di colore (TINTA- H)
mentre con RGB molte combinazioni di valori diversi da loro danno quel colore"""

def edge_detection(frame):  #CANNY
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    bordi = cv2.Canny(blur, 100, 200) #gradiente in modulo
    return bordi

"""1) soglia canny 
    2) immagine binaria sogliatura
    3) """

#-----------------------------------------ES1------------------------------------------------------
location = "externalFiles/solidWhiteCurve.jpg"
img = cv2.imread(location)

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(3, 2, 1).axis("off")
plt.imshow(imgRGB)
plt.title("immagine iniziale")

#SOGLIATURA- prendo righe bianche + gialle-----------------------------------------
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
"""PER IL BIANCO: 
prendo tutti i valori di TINTA + quasi tutti i valori di SATURAZIONE + un range di valori di luminosità
tutte le possibili sfumature , ma sempre di tinta bianca , in quanto esistono molti tipi di bianchi nell'img.
prendiamo mappati sul grafico HSV: per prendere il giallo prendo il settore angolare tra 20 e 30.
noi qui stiamo prendendo tutte le sfumature di tinta (per avere un bianco sbiadito), una saturazione data da
sensitività decisa e value relativa ad una certa sensitività."""
sensitivity = 30
sogliatura_bianco = cv2.inRange(imgHSV, np.array([0,0,255-sensitivity]), np.array([255,sensitivity,255])) #restituisce img solo con zone dove ho colori

"""PER UN COLORE NORMALE :
prende un pezzo dell'istogramma della TINTA (H) lasciando inalterata completamente la S E V cosi da prendere solo parte di esse"""
sogliatura_giallo = cv2.inRange(imgHSV, (20,100,100), (30,255,255))
# combine the mask
sogliaturaTOT = cv2.addWeighted(sogliatura_bianco, 1, sogliatura_giallo, 1, 0.0)
yw_mask = cv2.bitwise_or(sogliatura_bianco, sogliatura_giallo) #uguale

# apply the mask
sogliaturaTOT = cv2.bitwise_and(img, img, mask=yw_mask) #RESTITUISCE IMG BINARIA : mette 1 dove ho la corrispondenza con mask
#sogliaturaRGB = cv2.cvtColor(sogliaturaTOT, cv2.COLOR_BGR2RGB) #poiche la INRANGE fornisce img binaria ma in BGR

plt.subplot(3, 2, 2).axis("off")
plt.imshow(sogliaturaTOT)
plt.title("solidWhiteCurve-sogliatura")


#RILEVAZIONE EFFETTIVA DEI CONTORNI LINEE -----------------------------------------
#EDGE-DETECTION
bordi = edge_detection(sogliaturaTOT) #applichiamo Canny per bordare le linee
bordiRGB = cv2.cvtColor(bordi, cv2.COLOR_BGR2RGB)
plt.subplot(3, 2, 3).axis("off")
plt.imshow(bordiRGB)
plt.title("bordi1- Canny")

#DILATION
kernel = np.ones((2,2),np.uint8)
dilation = cv2.dilate(bordi,kernel,iterations = 2)
dilationRGB = cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB)
plt.subplot(3, 2, 4).axis("off")
plt.imshow(dilationRGB)
plt.title("dilation")

#HOUGH TRASFORMATION: RILEVAZIONE
linee = cv2.HoughLinesP(dilation,1,np.pi/180,100, minLineLength=10, maxLineGap=55)
print(linee) #return 4 valori per linea = [x1 y1 x2 y2] punti inizio e fine, è una lista con un solo elemento LISTA-> ogni linea in una LISTA
for line in linee:
    x1,y1,x2,y2 = line[0] #poichè line è una lista con unico elemento (ancora lista) line[0]
    print(line[0])
    cv2.line(imgRGB,(x1,y1),(x2,y2),(0,255,0),2) #disegno le linee sull'imgRGB
plt.subplot(3, 2, 5).axis("off")
plt.imshow(imgRGB)
plt.title("HOUGH")

#BLUR GAUSSIANO
blurG = cv2.GaussianBlur(img, (9, 9), 0)
blurGRGB = cv2.cvtColor(blurG, cv2.COLOR_BGR2RGB)
plt.subplot(3, 2, 6).axis("off")
plt.imshow(blurGRGB)
plt.title("immagine blur gaussiano")


plt.show()

