import cv2  # opencv
import numpy as np  # numpy

# create and print a 3x3 identity matrix on screen
# notice that 'Mat' is not available in OpenCV-Python, but numpy data structures are used
eye = np.eye(3, dtype="uint8")  # without 'uint8' it uses 'float' (default)
# alternatives:
# eye = np.eye(3, dtype=np.uint8)
# eye = np.eye(3, 3, 0, "uint8")
print("Identity matrix:", eye)

# read an image from disk
location = "images/Poli.jpg"
img = cv2.imread(location)

# convert the image in grayscale
print("Convert the image at '" + location + "' in grayscale...")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("... done!")

# save it on disk
cv2.imwrite("images/Poli-gray.jpg", gray)
