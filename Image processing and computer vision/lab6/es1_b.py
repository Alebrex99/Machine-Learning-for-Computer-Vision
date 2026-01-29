import cv2  # opencv
import numpy as np  # numpy
import matplotlib.pyplot as plt


def grab_frame(cap):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :return: the captured image
    """
    ret, frame = cap.read()  #ritorno frame + bool se letto
    sogliaturaTOT = sogliatura(frame)
    bordi = edge_detection(sogliaturaTOT)
    dilation = dilation_fun(bordi)
    houghRGB = hough_fun(dilation, frame)
    return houghRGB

def edge_detection(sogliaturaTOT):  #CANNY
    # EDGE-DETECTION
    blur = cv2.GaussianBlur(sogliaturaTOT, (9, 9), 0)
    bordi = cv2.Canny(blur, 100, 200)  # gradiente in modulo
    return bordi

def sogliatura(frame):
    # SOGLIATURA- prendo righe bianche + gialle-----------------------------------------
    imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    """PER IL BIANCO: 
    prendo tutti i valori di TINTA + quasi tutti i valori di SATURAZIONE + un range di valori di luminosità
    tutte le possibili sfumature , ma sempre di tinta bianca , in quanto esistono molti tipi di bianchi nell'img.
    prendiamo mappati sul grafico HSV: per prendere il giallo prendo il settore angolare tra 20 e 30.
    noi qui stiamo prendendo tutte le sfumature di tinta (per avere un bianco sbiadito), una saturazione data da
    sensitività decisa e value relativa ad una certa sensitività."""
    sensitivity = 30
    sogliatura_bianco = cv2.inRange(imgHSV, np.array([0, 0, 255 - sensitivity]),
                                    np.array([255, sensitivity, 255]))  # restituisce img BINARIA (1 è il sogliato)

    """PER UN COLORE NORMALE :
    prende un pezzo dell'istogramma della TINTA (H) lasciando inalterata completamente la S E V cosi da prendere solo parte di esse"""
    sogliatura_giallo = cv2.inRange(imgHSV, (20, 100, 100), (30, 255, 255))
    sogliaturaTOT = cv2.addWeighted(sogliatura_bianco, 1, sogliatura_giallo, 1, 0.0)
    #sogliaturaRGB = cv2.cvtColor(sogliaturaTOT, cv2.COLOR_BGR2RGB)
    return sogliaturaTOT

def dilation_fun(bordi):
    # DILATION
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(bordi, kernel, iterations=5)
    #dilationRGB = cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB)
    return dilation

def hough_fun(dilation, frame):
    # HOUGH TRASFORMATION: RILEVAZIONE
    imgRGB = frame
    linee = cv2.HoughLinesP(dilation, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=300)
    for line in linee:
        x1, y1, x2, y2 = line[0]
        cv2.line(imgRGB, (x1, y1), (x2, y2), (0, 255, 0), 2)  # disegno le linee sull'imgRGB
    return imgRGB


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
    :param image: the BGR image
    :return: the same image but in grayscale
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



def main():
    # init the camera
    cap = cv2.VideoCapture('externalFiles/solidWhiteRight.mp4')
    # enable Matplotlib interactive mode, finestra interattiva
    plt.ion()
    # create a figure to be updated, riquadro creato
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))


    # ACQUISISCO PRIMO FRAME
    ax_img = None
    while cap.isOpened():   #true se cattura inizializzata
        # get the current frame
        frame = grab_frame(cap) #richiama read
        if ax_img is None:
            # convert the current (first) frame in grayscale
            ax_img = plt.imshow(bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Video Capture")
            # show the plot!
            plt.show()
        else:
            #ACQUISISCO FRAME SUCCESSIVI
            ax_img.set_data(bgr_to_rgb(frame))
            # update the figure associated to the shown plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(1 / 30)  # pause: 30 frames per second







if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)


