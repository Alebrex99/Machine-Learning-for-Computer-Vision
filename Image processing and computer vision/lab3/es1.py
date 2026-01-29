import cv2  # opencv
import numpy as np  # numpy
import matplotlib.pyplot as plt

#FUNZIONE SENO ---------------------------------------------------------------------------
location = 'img/sinFunction.png'
seno = cv2.imread(location,0) #prende l'img
seno_complex = np.float32(seno) #trasformo immagine naturale in img complessa

#TRASFORMATA DFT
complex_image_result = cv2.dft(seno_complex, flags=cv2.DFT_COMPLEX_OUTPUT) #DFT
dft_shift = np.fft.fftshift(complex_image_result) #shift DFT
real_part_0 = complex_image_result[:,:,0] #real
imaginary_part_0 = complex_image_result[:,:,1] #imm

real_part_shift = dft_shift[:,:,0] #real
imaginary_part_shift = dft_shift[:,:,1] #imm
magnitude = 20*np.log(cv2.magnitude(real_part_shift,imaginary_part_shift)+1)

plt.subplot(4, 2, 1).axis("off") #img mostrerà range di pixel nell'img
plt.imshow(seno, "gray")
plt.title("immagine seno")
plt.subplot(4, 2, 2).axis("off") #img mostrerà range di pixel nell'img
plt.imshow(magnitude, "gray")
plt.title("modulo")

#ANTITRASFORMATA DFT
idft = cv2.idft(complex_image_result)
real_part = idft[:,:,0]
plt.subplot(4, 2, 3).axis("off") #img mostrerà range di pixel nell'img
plt.imshow(real_part, "gray")
plt.title("IDFT")



#----------------------------------CIRCOLINO ----------------------------------------------
location2 = 'img/circle.png'
circle = cv2.imread(location2,0) #prende l'img e converte anche in GRIGI
circle_complex = np.float32(circle) #trasformo immagine naturale in img complessa

#TRASFORMATA DFT
complex_image_result2 = cv2.dft(circle_complex, flags=cv2.DFT_COMPLEX_OUTPUT) #DFT
dft_shift2 = np.fft.fftshift(complex_image_result2) #shift DFT
real_part_0_2 = complex_image_result2[:,:,0] #real
imaginary_part_0_2 = complex_image_result2[:,:,1] #imm

real_part_shift2 = dft_shift2[:,:,0] #real
imaginary_part_shift2 = dft_shift2[:,:,1] #imm
magnitude2 = 20*np.log(cv2.magnitude(real_part_shift2,imaginary_part_shift2)+1)

plt.subplot(4, 2, 4).axis("off") #img mostrerà range di pixel nell'img
plt.imshow(circle, "gray")
plt.title("immagine circolo")
plt.subplot(4, 2, 5).axis("off") #img mostrerà range di pixel nell'img
plt.imshow(magnitude2, "gray")
plt.title("modulo")

#ANTITRASFORMATA DFT
idft2 = cv2.idft(complex_image_result2)
real_part2 = idft2[:,:,0]
plt.subplot(4, 2, 6).axis("off") #img mostrerà range di pixel nell'img
plt.imshow(real_part2, "gray")
plt.title("IDFT")

plt.show()


