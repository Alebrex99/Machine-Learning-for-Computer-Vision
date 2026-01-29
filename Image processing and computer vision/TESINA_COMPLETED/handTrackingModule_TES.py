"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""
import cv2
import mediapipe as mp
import time

"""OGGETTO HANDS -> per tracking mani"""

"""creo classe : 
    - costruttore : parametri di HANDS 
      oggetto creato : fornisco direttamente i parametri dell'utente: mode, maxHands , ecc.
    - metodi : 
              findHands -> trova le mani + disegna i landMarks se flag DRAW è true"""
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands)# NOTA : DA SISTEMARE, int(self.detectionCon), int(self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    """FINDHANDS -> trova le mani + disegna i landMarks se flag DRAW è true"""
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) #N.B. : Abilita processo riconoscimento MANI , dobbiamo estrarre le mani
        #print(self.results.multi_hand_landmarks)
        """per ogni punto estraggo info (FOR) , HANDLM = punti rossi sulla mano, cordinate x y z.
        cosi disegnamo i punti sulle mani
        usiamo metodo per disegnare linee collegamento punti -> MPDRAW : scriviamo su immagine BGR.
        troviamo 21 LANDMARKS (punti x y z).
        DOBBIAMO PRENDERE LE INFORMAZIONI DAI LANDMARKS"""
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                """usiamo una funzione landmark che estre info dai landmarks: indice ecc.
                   cosi stampiamo coordinate X Y Z; distinguo ID dal landmark completo
                   - x : val decimale in pixels è moltiplicato per high e weight; 
                   lo sistemiamo per avere significato in PIXELS:
                   H, W, C -> identifichiamo shape img 
                   CX, CY -> ogni pixel
                   successivamente disegnamo un circolo attorno al primo landmark (al pixel che lo riguarda)"""
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


    """FINDPOSITION -> trova pixels associati a handlandmarks + disegna cerchi se trova LANDMARKS
    dopo aver celto mano dx o sx con HANDNO, creo lista per ordinare i landmarks della mano scelta."""
    def findPosition(self, img, handNo = 0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return self.lmList

#CREO METODO PER VIRTUAL PAINT:
    def fingerUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]: #considera le X , che non sono invertite
            fingers.append(1) #aperto dito
        else:
            fingers.append(0)

        # 4 DITA: INDICE, MEDIO, ANULARE, MIGNOLO
        """MANO APERTA - CHIUSA :
        [id, cx, cy] : ogni elemento della lista, il secondo param prende l'elemento della sottolista, confrontiamo che la Y 
        del LANDMARK deve essere sotto la Y del LANDMARK, in tal modo indice alza falange (le Y dei pixels sono invertite)
        - se dito aperto -> creo una lista e metto valore 1 corrispondentemente al polpastrello aperto
        - se dito chiuso -> metto 0 nella stessa posizione
          [0 , 0 , 0 , 0, 0] -> [1, 0 , 0 , 0 , 0]"""
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1) #1 = aperto
            else:
                fingers.append(0)
        """molto utile per tornare la lista fingers che dice chi è aperto e chi chiuso tra le dita"""
        return fingers

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        #print(lmList)
        """se non vede nessuna mano metto if per evitare errore, cosi stampa solo se rintraccia i LANDMARKS"""
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
   main()