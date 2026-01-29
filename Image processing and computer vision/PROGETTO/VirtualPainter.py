import cv2
import numpy as np
import time
import os
import handTrackingModule_InProgress as htm

###########COSTRUZIONE VIEWPORT + VIRTUAL PAINTER############
brushThickness = 25
eraserThickness = 100
########################


folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
"""immagini che vogliamo porre nella view port:"""
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
header = overlayList[0] #questo è header della viewport, parte alta -> SONO LE IMMAGINI CHE RAPPRESENTANO I PENNELLI
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

"""fondamentale: impostare il detector mano con maggior precisione : 0,65 , piuttosto che default 0,5"""
detector = htm.handDetector(detectionCon=0.65,maxHands=1)
xp, yp = 0, 0
"""costruisco un canvas in modo disegnare su di esso  """
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


while True:
    #INDICE :
    #1. import image
    #2. trovare i landmarks mano
    #3. controllo se dita sono su
    #4. if selection mode
    """operazioni preliminari: 
    slice immagine frame, scegli la parte dove mettere header"""

    # 1. Import image--------------------------------------------------------------
    success, img = cap.read()
    """bisogna flippare la schermata dx-> sx per poter scrivere a mano su shcermo"""
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks-------------------------------------------------------
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False) #ricorda che draw serve per disegnar ei cerchietti sui landmarks

    if len(lmList) != 0:
        # print(lmList)
        # posizioni di INDICE E MEDIO fingers -> polpastrelli
        x1, y1 = lmList[8][1:] #INDICE
        x2, y2 = lmList[12][1:]#MEDIO

        # 3. Check which fingers are up--------------------------------------------------
        fingers = detector.fingersUp()
        # print(fingers)

        """se 2 dita sono su : SELECTION mode , altrimenti DRAWING mode
        SELECTION MODE -> disegna un rettangolo tra le due dita
        DRAWING MODE -> disegna un cerchio tra le dita"""

        # 4. SELECTION MODE : If Selection Mode – Two finger are up---------------------------------------------------
        """indice + medio"""
        if fingers[1] and fingers[2]:
            # xp, yp = 0, 0
            print("Selection Mode")
            #POSIZIONE SU HEADER : controllo posizione dita in HEADER
            if y1 < 125:
                """le X identificano i vari pennelli (immagini) nell'HEADER + scegliamo il colore del pennello
                ogni volta che passo su un pennello sostituisco l'header con immagine di un pennello """
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
                    cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5.DRAWING MODE : If Drawing Mode – Index finger is up---------------------------------------------------------
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            # if drawColor == (0, 0, 0):
            #   cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
            #   cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            #
            # else:
            #   cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            #   cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    # # Clear Canvas when all fingers are up
    # if all (x >= 1 for x in fingers):
    # imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    #COME DISEGNARE: USARE CANVAS SOVRAPPOSTO
    """prendi il disegno sul CANVAS, lo converti in img binaria"""
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv) #and tra img e canvas: parti a 0 nel canvas , pongono a 0 il frame acquisito, cosi creo la linea del disegno
    img = cv2.bitwise_or(img,imgCanvas) #se il canvas si riempie, rimepi anche l'img

    # Setting the header image
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)