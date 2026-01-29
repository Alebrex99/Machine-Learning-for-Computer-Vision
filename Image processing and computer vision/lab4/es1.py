import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np


def edge_detection(frame): #CANNY
    blur = cv2.GaussianBlur(frame, (5, 5), 0) #in realtà è gia previsto da algoritmo CANNY
    bordi = cv2.Canny(blur, 0, 200) #dentro la soglia ho px connessi; se aumenti delta tra le soglie
    #a contrasto piu alto nelle immmagini avrai meno px connessi, che vengon buttagti a 0
    return bordi

def edge_detection_sobel(frame):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    window_name = ('Sobel- Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def grab_frame(cap):
    """
    Method to grab a frame from the camera
    :param cap: the VideoCapture object
    :return: the captured image
    """
    ret, frame = cap.read()  #ritorno frame + bool se letto
    frame_gaussian_sobel = edge_detection_sobel(frame) #applico gaussian blur + sobel
    frame_gaussian_Canny = edge_detection(frame)
    return frame_gaussian_sobel

def add_logo(frame):
    location = "images/logo_poli.jpg"
    img = cv2.imread(location)
    #metodo 1
    if img is not None :
        print(img.shape)  # 194, 194 (logo)    ;    480, 640 (frame)
        spazio = tuple(numpy.subtract(frame.shape, img.shape)) #286 * 446
        frame[spazio[0]:frame.shape[0], spazio[1]: frame.shape[1]] = img
    return frame




def handle_close(event, cap):
    """
    Handle the close event of the Matplotlib window by closing the camera capture
    :param event: the close event
    :param cap: the VideoCapture object to be closed
    """
    cap.release()


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


    # ACQUISISCO PRIMO FRAME
    ax_img = None
    while cap.isOpened():   #true se cattura inizializzata
        # get the current frame
        frame = grab_frame(cap) #richiama read
        #print(frame.shape) # 640*480 il frame è la grandezza della finestra, vuota
        if ax_img is None:
            # convert the current (first) frame in grayscale
            ax_img = plt.imshow(bgr_to_rgb(frame), "gray")
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
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
