import cv2
import matplotlib.pyplot as plt


def grab_frame(cap):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :return: the captured image
    """
    ret, frame = cap.read()
    return frame


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


def main():
    # init the camera
    cap = cv2.VideoCapture(0) #attiva la webcam di 0 , la prima del PC

    # enable Matplotlib interactive mode
    plt.ion() #abilita un modo interattivo di matplot lib , in modo di rapp flussi video, piu img

    # create a figure to be updated : contenitore.
    fig = plt.figure()
    #creo figura, infatti quando faccio plot/img show viene creata una figura :
    #prendo riferimento esplicito alla figura (fig) a cui collego un evento di chiusura della finestra e un metodo
    #che intercetta EVENTO (close_event):
    #quando hciudiamo la finestra di matplot lib (per img) rilascio e chiudo webcam , altrimenti programma rimane in exe e webcam continua a funzionare.
    #recupero figura principale generata con imshow (tramite figure()), a tale figura agganciamo
    #evento chiusura finestra (X finestra), ossia funzione HANDLE_CLOSE (funzione legata ad evento):
    # - cap.release() ; rilascio chiusura webcam
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))

    # prep a variable for the first run:
    #serve come supporto (matplot lib funziona con img singoli) devo avere var supoorto che si comporta
    #in base a se è primo frame flusso video; se è secondo frame aggiorno la variabile se no rimpiazzerebbe img alla volta
    ax_img = None

    while cap.isOpened():
        # get the current frame
        frame = grab_frame(cap)
        if ax_img is None:
            # convert the current (first) frame in grayscale
            ax_img = plt.imshow(bgr_to_gray(frame), "gray")
            #genero con IMSHOW una finestra, sparo il dato frame, e salvo contenuto
            #puntatore posto in ax_img che
            #dovro visualizzare con SHOW, il contenuto puntato da ax_img
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
            # show the plot!
            plt.show()
            print("primo frame")
        else:
            # set the current frame as the data to show
            print("frame successivi")
            ax_img.set_data(bgr_to_gray(frame))  #prende riferimento alla finestra e aggiorna con set_data
            # update the figure associated to the shown plot
            fig.canvas.draw()  #disegna effettivamente la finestra che ora vediamo aggiornata
            fig.canvas.flush_events()
            plt.pause(1/30)  # pause: 30 frames per second


#tutto cio scritto nel main, eseguilo solo quando file python lanciato direttamente, se file python importato come modulo in un altro file
#questo non viene visto, non si puo accedere a cio scritto in questo main, cio scritto (se faccio import) visibile come modulo.
if __name__ == "__main__": #se lancia un eccezione esci ed interrompi flusso video
    try:
        main()
    except KeyboardInterrupt: #interruzione flusso video tramite tastiera
        exit(0)
