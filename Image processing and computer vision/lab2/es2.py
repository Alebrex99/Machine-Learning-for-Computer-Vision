import cv2
import matplotlib.pyplot as plt
import numpy as np


def grab_frame(cap, img):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :return: the captured image
    """
    ret, frame = cap.read()  #ritorno frame + bool se letto
    add_logo(frame, img)
    return frame

def add_logo(frame, img):
    """location = "images/logo_poli.jpg"
    img = cv2.imread(location)"""
    #metodo 1 - intera sostituzione pixels
    """
    if img is not None :
        print(img.shape)  # 194, 194 (logo)    ;    480, 640 (frame)
        spazio = tuple(numpy.subtract(frame.shape, img.shape)) #286 * 446
        frame[spazio[0]:frame.shape[0], spazio[1]: frame.shape[1]] = img
    return frame
    """


    #metodo 2 - con Blending
    spazio = frame[286:480, 446:640] #pezzo da pesare
    #logo_finale = np.zeros((194,194))
    cv2.addWeighted(spazio, 0.4, img, 0.6, 0.0, spazio)
    #frame[286:480 , 446:640] = logo_finale
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

def bgr_to_rgb(image):
    """
    Convert a BGR image into grayscale
    :param image: the BGR image
    :return: the same image but in grayscale
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



def main():
    # init the camera
    cap = cv2.VideoCapture(0) #open video stream , inizializzo
    # enable Matplotlib interactive mode, finestra interattiva
    plt.ion()
    # create a figure to be updated, riquadro creato
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))

    location = "images/logo_poli.jpg"
    img = cv2.imread(location)

    # ACQUISISCO PRIMO FRAME
    ax_img = None
    while cap.isOpened():   #true se cattura inizializzata
        # get the current frame

        frame = grab_frame(cap, img) #richiama read
        #print(frame.shape) # 640*480 il frame è la grandezza della finestra, vuota
        if ax_img is None:
            # convert the current (first) frame in grayscale
            ax_img = plt.imshow(bgr_to_rgb(frame), "gray") #bgr_to_gray(frame)
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
            # show the plot!
            plt.show()
        else:
            #ACQUISISCO FRAME SUCCESSIVI
            ax_img.set_data(bgr_to_rgb(frame)) #bgr_to_gray(frame)
            # update the figure associated to the shown plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(1 / 30)  # pause: 30 frames per second







if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
