import cv2
import time
import os
import handTrackingModule_InProgress as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(1)
cap.set(3, wCam) #prendo la larghezza e altezza camera e le setto
cap.set(4, hCam)

"""prendiamo os.listdir che prende tutte le immagini in una directory
le inserisco in una lista
OVERLAYLIST : è una lista vuota """
folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

"""per ogni img, la leggo, prendo l'obj restituito da imread e lo inserisco in OVERLAYLIST"""
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
pTime = 0


"""creiamo l'obj DETECTOR ,(modulo separato), cro una istanza di tale classe;
   TIPIDS : sono specificato i LANDMARKS dei polpastrelli: generalizzo codice"""
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img) #metodo dell'altro modulo hand_detector, disegno punti e landmark su mano
    lmList = detector.findPosition(img, draw=False) #metodo restituisce la LISTA di tutti e 21 i punti rintracciati (landmarks) + disegna cerchi
    # print(lmList)

    """CODICE DELLA CONTA DELLE DITA:-------------------------------------------------------------------
    parte codice che verrà generalizzata"""
    if len(lmList) != 0:
        """[id, cx, cy] : ogni elemento della lista, il secondo param prende l'elemento della sottolista, confrontiamo che la Y 
        del LANDMARK 8 deve essere sotto la Y del LANDMARK 6, in tal modo indice abbassa falange"""
        if lmList[8][2] < lmList[6][2]:
            print("Index finger open")

    if len(lmList) != 0:
        fingers = []
        # POLLICE
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]: #considera le X , che non sono invertite
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 DITA: INDICE MEDIO ANULARE MIGNOLO
        """MANO APERTA - CHIUSA :
        [id, cx, cy] : ogni elemento della lista, il secondo param prende l'elemento della sottolista, confrontiamo che la Y 
        del LANDMARK deve essere sotto la Y del LANDMARK, in tal modo indice alza falange (le Y dei pixels sono invertite)
        - se dito aperto -> creo una lista e metto valore 1 corrispondentemente al polpastrello aperto
        - se dito chiuso -> metto 0 nella stessa posizione
          [0 , 0 , 0 , 0, 0] -> [1, 0 , 0 , 0 , 0]"""
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1) #1 = aperto
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1) #ritorna quante occorrenze della lista hanno 1
        print(totalFingers)

        """le img piccole sono 200*200 , vanno inserite come slice, prendo altezza profondità e canale."""
        """img[100:300, 100:300] = overlayList[0] #slice dell'img , cosi posso rapp le immagini sulla webcam con slice"""
        h, w, c = overlayList[totalFingers - 1].shape #per selezionare l'img
        img[0:h, 0:w] = overlayList[totalFingers - 1] #ricorda che lista[-1] -> prende l'ultimo elemento , quindi quando fingercount =0 , prende 6.jpg

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    """la libreria time() con currentTime e previousTime servono per stampare a video gli FPS"""
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)