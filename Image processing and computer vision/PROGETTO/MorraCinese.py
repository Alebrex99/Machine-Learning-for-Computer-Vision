import cv2
import matplotlib.pyplot as plt
import time
import os
import handTrackingModule_InProgress as htm
from enum import Enum
import random

wCam, hCam = 1280, 720
tipIds = [4, 8, 12, 16, 20] #polpastrelli
currentGesture = -1 # -1 default, 0 sasso, 1 carta, 2 forbice

class State(Enum):
    _Menu = 1
    _Game = 2
    _Result = 3

currentState = State._Menu

startPointButtonStart = (400, 16)
endPointButtonStart = (880, 240)

cTime=0
pTime=0
tTime = 0
"""FINGERS 4 ELEMENTI INDICE-MEDIO-ANULARE-MIGNOLO"""
"""LM LIST : [id, cx, cy]"""

def gestureDetection(lmList, img):
    global currentState
    global tTime, cTime, pTime
    global playerChoice, pcChoice
    cState = currentState

    if len(lmList) != 0:
        fingers = []
        # rilevo mano orientata verso l'alto/basso
        if lmList[9][2] < lmList[0][2]:
            # 4 DITA: INDICE MEDIO ANULARE MIGNOLO
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            #orientation = "alto"
        else:
            for id in range(1, 5):
                if lmList[tipIds[id]][2] > lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            #orientation = "basso"

        # MODIFICA : SPOSTAMENTO ASSEGNAZIONI CURRENT GESTURE-------------------------------


        # CHECK STATE
        if currentState == State._Menu:
            x1, y1 = lmList[tipIds[1]][1:]
            if fingers[0]:  # and fingers[2]
                # POSIZIONE SU Start
                if y1 < endPointButtonStart[1]:
                    if startPointButtonStart[0] < x1 < endPointButtonStart[0]:
                        cTime = 0
                        pTime = 0
                        tTime = 0
                        cState = State._Game
        elif currentState == State._Game:
            if tTime > 3:
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
                tTime = 0
                playerChoice = currentGesture
                pcChoice = random.randint(0, 2)
                cTime = 0
                pTime = 0
                tTime = 0
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

    img = detector.findHands(frame)  # metodo dell'altro modulo hand_detector, disegno punti e landmark su mano
    lmList = detector.findPosition(img)#, draw=False
    gestureDetection(lmList, img)
    # update state
    if currentState == State._Menu:
        cv2.rectangle(img, startPointButtonStart, endPointButtonStart, (0, 0, 255), 1)
        #print("Menu")

    elif currentState == State._Game:
        cv2.circle(img, (640, 360), 180, 5)
        cTime = time.time()
        if pTime == 0:
            pTime = cTime
        tTime = tTime + (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(tTime)), (640, 360), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 3)
        if tTime > 3:
            cv2.putText(img, "Mano non rilevata", (340, 400), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 3)
       # print("Game")

    elif currentState == State._Result:
        # GENERO NUMERO RANDOM -> SASSO CARTA FORBICE
        # TEMPI TUTTI A 0
        # LEGGO LA VARIABILE GLOBALE CURRENT GESTURE
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


def main():
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
        # get the current frame
        frame = grab_frame(cap, detector)
        if ax_img is None:
            # convert the current (first) frame in grayscale
            ax_img = plt.imshow(bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
            # show the plot!
            plt.show()
        else:
            # set the current frame as the data to show
            ax_img.set_data(bgr_to_rgb(frame))
            # update the figure associated to the shown plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(1/30)  # pause: 30 frames per second


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
