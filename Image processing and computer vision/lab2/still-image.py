import cv2
import matplotlib.pyplot as plt

# prepare to convert a RGB image in grayscale and save it
location = 'images/Poli.jpg'
img = cv2.imread(location)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# now, we want to show the two images...
# convert in RBG for matplotlib and prepare the first plot
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(1, 2, 1).axis('off')  # subplot; plus, it hides ticks and borders
plt.imshow(img)
plt.title("RGB Image")

# convert in grayscale and show both plots
plt.subplot(1, 2, 2).axis('off')
plt.imshow(gray, 'gray')  # or cmap='gray'
plt.title("Grayscale Image")
plt.show()
