"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""
import cv2
import mediapipe as mp
import time

"""OGGETTO HANDS
"""

cap =cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
"""inserisco gli fps , definisco"""
pTime = 0
cTime = 0


while True:
    """uso codice LAB"""
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) #N.B. : Abilita processo riconoscimento MANI , dobbiamo estrarre le mani
    #print(results.multi_hand_landmarks)

    """per ogni punti estraggo info (FOR) , HANDLM = punti rossi sulla mano, cordinate x y z.
    cosi disegnamo i punti sulle mani
    usiamo metodo per disegnare linee collegamento punti -> MPDRAW : scriviamo su immagine BGR.
    troviamo 21 LANDMARKS (punti x y z).
    DOBBIAMO PRENDERE LE INFORMAZIONI DAI LANDMARKS"""
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            """usiamo una funzione landmark che estre info dai landmarks: indice ecc.
               cosi stampiamo coordinate X Y Z; distinguo ID dal landmark completo
               - x : val decimale in pixels è moltiplicato per high e weight; 
               lo sistemiamo per avere significato in PIXELS:
               H, W, C -> identifichiamo shape img 
               CX, CY -> ogni pixel
               successivamente disegnamo un circolo attorno al primo landmark (al pixel che lo riguarda)"""
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                #if id==4:
                cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    """definisco il tempo corrente e precedente per decidere il frame rate + Lo visualizzo su schermo con PUTTEXT
    """
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1) #delay della camera