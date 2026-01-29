import cv2
import matplotlib.pyplot as plt
import time
import os
import handTrackingModule_TES as htm
from enum import Enum
import random

#VARIABILI GLOBALI
wCam, hCam = 1280, 720
tipIds = [4, 8, 12, 16, 20] #polpastrelli
currentGesture = -1 # -1 default, 0 sasso, 1 carta, 2 forbice
global screenShot

# CARICAMENTO IMMAGINI
menuImg = cv2.imread("img/menuImg.png")
menuImg= cv2.resize(menuImg, (1280, 720))
resultImg = cv2.imread("img/cartaa.png")
resultImg= cv2.resize(resultImg, (1280, 720))

class State(Enum):
    _Menu = 1
    _Game = 2
    _Result = 3

currentState = State._Menu

startPointButtonStart = (35, 25)
endPointButtonStart = (385, 280)


startPointButtonTutorial = (1115, 65)
endPointButtonTutorial = (1208, 155)

cTime=0
pTime=0
tTime = 0
"""FINGERS 4 ELEMENTI INDICE-MEDIO-ANULARE-MIGNOLO"""
"""ID NOCCHE = 5 9 13 17
   LM LIST : [id, cx, cy], ...-> es) [[0, 435, 591], [1, 491, 618], ...]
   TIPID : tipIds = [4, 8, 12, 16, 20] #polpastrelli
   FINGERS : [0,1,0,1] - si aggiorna ad ogni grab_frame() -> 1 su, 0 giù : relativi alle dita = 8, 12, 16, 20"""

def gestureDetection(lmList, img, screenShot):
    global currentState
    global tTime, cTime, pTime
    global playerChoice, pcChoice
    cState = currentState
    if len(lmList) != 0:
        fingers = []
        # rilevo mano orientata verso l'ALTO (nocca sopra polsa)
        if lmList[9][2] < lmList[0][2]:
            #rilevo dito aperto/chiuso (riempio fingers[])
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        # rilevo mano orientata verso l'BASSO (nocca sotto polso)
        else:
            #rilevo dito aperto/chiuso (riempio fingers[])
            for id in range(1, 5):
                if lmList[tipIds[id]][2] > lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            #orientation = "basso"

        # -------------------------MODIFICA : SPOSTAMENTO ASSEGNAZIONI CURRENT GESTURE-------------------------------


        # CHECK STATE--------quando cambio stato-----------
        if currentState == State._Menu:
            x1, y1 = lmList[tipIds[1]][1:]
            x2, y2 = lmList[tipIds[2]][1:]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # controllo indice alzato
            if fingers[0] and fingers[1]:
                # indice su riquadro rosso=Start
                if cy < endPointButtonStart[1]:
                    if startPointButtonStart[0] < cx < endPointButtonStart[0]:
                        cTime = 0
                        pTime = 0
                        tTime = 0
                        cState = State._Game

        elif currentState == State._Game:
            if tTime > 3 and passoUnaVolta:
                passoUnaVolta = False
                print("scatto uno screenshot !")
                #------------------------------------------------
                # check gesture # DA SPOSTARE ALL'INTERNO DI GAME -> OUTPUT CURRENT GESTURE LETTO DAL GRABFRAME
                if fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
                    currentGesture = 0
                    # print("sasso" + orientation)
                elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                    currentGesture = 1
                    # print("carta" + orientation)
                elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                    currentGesture = 2
                    # print("forbice" + orientation)
                #----------------------------------------------------
                #tTime = 0
                screenShot = img
                playerChoice = currentGesture
                pcChoice = random.randint(0, 2)
                #cTime = 0
                #pTime = 0
                #tTime = 0
            if tTime > 6:
                tTime = 0
                cTime = 0
                pTime = 0
                tTime = 0
                passoUnaVolta = True
                cState = State._Result
        elif currentState == State._Result:
            if tTime == 0:
                print("player: ", playerChoice)
                print("pc: ", pcChoice)
                if playerChoice == pcChoice:
                    print("PAREGGIO")
                elif playerChoice == 0 and pcChoice == 2 or playerChoice == 1 and pcChoice == 0 or playerChoice == 2 and pcChoice == 1:
                    print("HAI VINTO")
                else:
                    print("HAI PERSO")
                # DOBBIAMO IMPLEMENTARE UN BOTTONE PER RICOMINCIARE
                print("-")

            if tTime > 3:
                cState = State._Menu

        currentState = cState

def grab_frame(cap, detector):
    global tTime, cTime, pTime
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :return: the captured image
    """
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    img = detector.findHands(frame)  # metodo dell'altro modulo hand_detector, disegno punti e landmark su mano
    lmList = detector.findPosition(img)#draw=False
    screenShot = img
    gestureDetection(lmList, img, screenShot)

    # UPDATE STATE-------------------------------------------------
    if currentState == State._Menu:
        #AGGIUNTA IMMAGINE MENU + PUNTATORE SU INDICE
        add_immagineMenu(frame)
        if len(lmList) != 0:
            x1, y1 = lmList[tipIds[1]][1:]
            x2, y2 = lmList[tipIds[2]][ 1:]
            cx,cy = (x1+x2) //2, (y1+y2) //2
            if abs(x2-x1)<= 90 and abs(y2-y1)<= 110:
                cv2.circle(img, (cx, cy), 40, (0, 255, 0), cv2.FILLED)
            print(x2-x1)
        #IMG START
        #cv2.rectangle(img, startPointButtonStart, endPointButtonStart, (0, 0, 255), 5, cv2.LINE_AA)
        #cv2.rectangle(img, startPointButtonTutorial, endPointButtonTutorial, (0, 0, 255), 5, cv2.LINE_AA)
        #cv2.putText(img, 'START GAME', (450, 160), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)

    elif currentState == State._Game:
        cv2.circle(img, (640, 360), 180, (0, 255, 0), 15)
        cTime = time.time()
        if pTime == 0:
            pTime = cTime
        tTime = tTime + (cTime - pTime)
        pTime = cTime
        textsize = cv2.getTextSize(str(int(tTime)), cv2.FONT_HERSHEY_DUPLEX, 10, 10)
        #print('textsize= ', textsize)
        textX = int(wCam / 2 - (int(textsize[0][0]) / 2))
        textY = int(hCam / 2 + (int(textsize[0][1]) / 2))
        cv2.putText(img, str(int(tTime)), (textX, textY), cv2.FONT_HERSHEY_TRIPLEX, 10, (0, 255, 0), 10)
        if tTime > 3:
            img = screenShot
            cv2.putText(img, "Mano non rilevata", (340, 400), cv2.FONT_HERSHEY_TRIPLEX, 10, (255, 0, 0), 3)
        # print("Game"

    elif currentState == State._Result:
        # GENERO NUMERO RANDOM -> SASSO CARTA FORBICE
        # TEMPI TUTTI A 0
        # LEGGO LA VARIABILE GLOBALE CURRENT GESTURE
        add_immagineResult(frame)
        cTime = time.time()
        if pTime == 0:
            pTime = cTime
        tTime = tTime + (cTime - pTime)
        pTime = cTime
        #print("Result")

    return img


def handle_close(event, cap):
    """
    Handle the close event of the Matplotlib window by closing the camera capture
    :param event: the close event
    :param cap: the VideoCapture object to be closed
    """
    cap.release()

def bgr_to_gray(image):
    """
    Convert a BGR image into grayscale
    :param image: the BGR image
    :return: the same image but in grayscale
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def bgr_to_rgb(image):
    """
    Convert a BGR image into grayscale
    :param image: the RGB image
    :return: the same image but in RGB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



def add_immagineMenu(frame):
    #if currentState ==State._Menu:
    cv2.addWeighted(frame, 0,menuImg , 1, 0.0, frame)
    return frame

def add_immagineResult(frame):
    cv2.addWeighted(frame, 0, resultImg, 1, 0.0, frame)
    #cv2.putText(frame, "RESULT", (640, 360), cv2.FONT_HERSHEY_TRIPLEX, 10, (255, 0, 0), 3)





def captureModeMurtaza():
    pTimeFps = 0
    cTimeFps = 0
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector()
    while True:
        frame = grab_frame(cap, detector)

        cTimeFps = time.time()
        fps = 1 / (cTimeFps - pTimeFps)
        pTimeFps = cTimeFps
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", frame)
        cv2.waitKey(1)


def main():
    #captureModeMurtaza()

    #FPS
    pTimeFps = 0
    cTimeFps = 0

    # init the camera
    cap = cv2.VideoCapture(0)

    cap.set(3, wCam)
    cap.set(4, hCam)

    # enable Matplotlib interactive mode
    plt.ion()

    # create a figure to be updated
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))

    # prep a variable for the first run
    ax_img = None

    detector = htm.handDetector()# DA AGGIORNARE detectionCon=0.75

    while cap.isOpened():
        # FPS
        cTimeFps = time.time()
        fps = 1 / (cTimeFps - pTimeFps)
        pTimeFps = cTimeFps
        # get the current frame
        frame = grab_frame(cap, detector)

        if ax_img is None:
            # convert the current (first) frame in grayscale
            cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)

            ax_img = plt.imshow(bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
            # show the plot!
            plt.show()
        else:
            #FPS
            cTimeFps = time.time()
            fps = 1 / (cTimeFps - pTimeFps)
            pTimeFps = cTimeFps
            cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)
            # set the current frame as the data to show
            ax_img.set_data(bgr_to_rgb(frame))
            # update the figure associated to the shown plot
            #fig.canvas.draw()
            #fig.canvas.flush_events()
            plt.pause(1/30)  # pause: 30 frames per second

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
