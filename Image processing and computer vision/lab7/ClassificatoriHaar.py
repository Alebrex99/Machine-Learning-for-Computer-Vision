import cv2
import matplotlib.pyplot as plt
import argparse

locationHaar = 'haarcascades/haarcascade_frontalface_default.xml'
#xml -> è un modello nato eseguendo l'algoritmo di classificazione Hair: ossia tramite metodo a cascata esso è stato allenato

locationLbp = 'lbpcascades/lbpcascade_frontalface.xml'

face_cascade = cv2.CascadeClassifier(locationHaar)
#dato il modello , costruito con l'algoritmo, lo diamo in pasto al classificatore cosi che riconosca
#quando c'è una faccia, diversa da altri oggetti.
#1) classificatore di Hair (implementato con Cascade)
#2) PREPROCESSING: modello creato con training : ottengo features di Haar (edge features, line features ecc)
#3) do un img (frame video) : riconosce

#eyes_cascade = cv2.CascadeClassifier()

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray) #memorizza tutti i volti in insieme di tuple faces
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        #eyes = eyes_cascade.detectMultiScale(faceROI)
        #for (x2,y2,w2,h2) in eyes:
           # eye_center = (x + x2 + w2//2, y + y2 + h2//2)
           # radius = int(round((w2 + h2)*0.25))
           # frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

    return frame

def edge_detection_canny(frame):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    return edges

def edge_detection_sobel(frame):
    ddepth = cv2.CV_16S
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    blur_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(blur_gray, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(blur_gray, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
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
    ret, frame = cap.read()
    #frame_canny = edge_detection_canny(frame)  #CANNY
    frame_sobel = edge_detection_sobel(frame)  #SOBEL
    return detectAndDisplay(frame)


def handle_close(event, cap):
    """
    Handle the close event of the Matplotlib window by closing the camera capture
    :param event: the close event
    :param cap: the VideoCapture object to be closed
    """
    cap.release()


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    # init the camera
    cap = cv2.VideoCapture(0)

    # enable Matplotlib interactive mode
    plt.ion()

    # create a figure to be updated
    fig = plt.figure()
    # intercept the window's close event to call the handle_close() function
    fig.canvas.mpl_connect("close_event", lambda event: handle_close(event, cap))

    # prep a variable for the first run
    img = None

    # read and store the logo
    logo = cv2.imread("img/logo.png")

    while cap.isOpened():
        # get the current frame
        frame = grab_frame(cap)
        if img is None:
            # convert it in RBG (for Matplotlib)
            img = plt.imshow(bgr_to_rgb(frame))
            plt.axis("off")  # hide axis, ticks, ...
            plt.title("Camera Capture")
            # show the plot!
            plt.show()
        else:
            # set the current frame as the data to show
            img.set_data(bgr_to_rgb(frame))
            # update the figure associated to the shown plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(1/30)  # pause: 30 frames per second


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)