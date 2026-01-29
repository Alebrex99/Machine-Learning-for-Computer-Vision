import cv2  # opencv
import numpy as np  # numpy
import matplotlib.pyplot as plt

location = 'img/tramonto.jpg'
tramonto = cv2.imread(location,0) #prende l'img e converte anche in GRIGI
tramonto_complex = np.float32(tramonto) #trasformo immagine naturale in img complessa

plt.subplot(2, 2, 1).axis("off") #img mostrerà range di pixel nell'img
plt.imshow(tramonto, "gray")
plt.title("immagine Tramonto")

#TRASFORMATA DFT
complex_image_result = cv2.dft(tramonto_complex, flags=cv2.DFT_COMPLEX_OUTPUT) #DFT
dft_shift = np.fft.fftshift(complex_image_result) #shift DFT

real_part_shift = dft_shift[:,:,0] #real shift
imaginary_part_shift = dft_shift[:,:,1] #imm shift
magnitude = 20*np.log(cv2.magnitude(real_part_shift,imaginary_part_shift)+1) #il + 1 serve per viasualizzarla

plt.subplot(2, 2, 2).axis("off") #img mostrerà range di pixel nell'img
plt.imshow(magnitude, "gray")
plt.title("modulo")


#CREAZIONE MASCHERA/FILTRO PASSA BASSO
print(tramonto.shape)
rows, cols = tramonto.shape
crow, ccol = int(rows / 2), int(cols / 2)
mask = np.ones((rows, cols, 2), np.uint8)
r1 = 80
r2 = 120
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
#cocordinate riga col : prende da 0 a rows, , a:cols e genera array
#di 2 elem primo array colonna e secondo array riga : finiscono X= ARRAY con 0000, 1111, 2222
mask_area = np.logical_and( ((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r1*r1),
                            (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r2*r2)
mask[mask_area] = 0
tramonto_filtrato = dft_shift*mask

f_ishift = np.fft.ifftshift(tramonto_filtrato)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(2, 2, 3).axis("off") #img mostrerà range di pixel nell'img
plt.imshow(img_back, "gray")
plt.title("IDFT")

plt.show()