from sys import argv
import cv2
_debug = False
if _debug:
    import matplotlib.pyplot as plt

if len(argv) != 3 and len(argv) != 4:
    print("args: metalness, roughness, [-s]")
    print("usa -s solo se hai già una mappa smoothness invece di una roughness")
    exit(-1)
smooth = False
if len(argv) == 4 and argv[3] == '-s':
    smooth = True

met = cv2.imread(argv[1])
smt = cv2.imread(argv[2])
out = met.copy()
out = cv2.cvtColor(out, cv2.COLOR_RGB2RGBA)
met = cv2.cvtColor(met, cv2.COLOR_RGB2GRAY)
smt = cv2.cvtColor(smt, cv2.COLOR_RGB2GRAY)

if not smooth:
    smt = cv2.bitwise_not(smt)                                      # converto roughness in smoothness
out[:, :, 2] = met                                                  # metto metalness in canale R
out[:, :, 1] = 0                                                    # azzero canale G
out[:, :, 0] = 0                                                    # azzero canale B
out[:, :, 3] = smt                                                  # metto smoothness in canale A

cv2.imwrite("out.png", out)                                         # salvo immagine

# test output
if _debug:
    plt.subplot(1, 3, 1).axis("off")
    plt.imshow(met, 'gray')
    plt.title("Metallic")

    plt.subplot(1, 3, 2).axis('off')
    plt.imshow(smt, 'gray')
    plt.title("Smoothness")

    out = cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA)
    plt.subplot(1, 3, 3).axis('off')
    plt.imshow(out)
    plt.title("Combined")
    plt.show()
