import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_lines(image, lines):
    copied_img = np.copy(image)  # don't want to modify the original
    for l in lines:
        for x1, y1, x2, y2 in l:
            cv2.line(copied_img, (x1, y1), (x2, y2), [0, 0, 255], 7)
    copied_img = cv2.cvtColor(copied_img, cv2.COLOR_BGR2RGB)
    return copied_img


def main():
    img = cv2.imread("externalFiles/solidWhiteCurve.jpg")

    # move to HSL, since yellow and white lines are well distinguished
    img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # debug
    # plt.imshow(img_hsl)

    # threshold to keep yellow and white lines
    # white color mask, only high Light value
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([179, 255, 255])
    white_mask = cv2.inRange(img_hsl, lower, upper)

    plt.subplot(2, 2, 1).axis("off")
    plt.imshow(white_mask)
    plt.title("maschera bianca")

    # yellow color mask, Hue around 30 and relatively high Saturation, all Light values
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(img_hsl, lower, upper)
    # combine the mask
    yw_mask = cv2.bitwise_or(white_mask, yellow_mask)
    # apply the mask
    masked_img = cv2.bitwise_and(img, img, mask=yw_mask)

    plt.subplot(2, 2, 2).axis("off")
    plt.imshow(masked_img)
    plt.title("immagine mascherata")

    # go grayscale and blur (for Canny)
    gray = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(gray, 50, 100)  # ratio 2:1 or 3:1

    # select a region of interest
    # first, define the polygon by vertices
    rows, cols = edges.shape[:2]
    bottom_left = [cols*0.1, rows*0.95]
    top_left = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right = [cols*0.6, rows*0.6]
    # the vertices are an array of polygons (i.e., array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    # defining a blank mask to start with
    mask = np.zeros_like(edges)

    # defining a 3 channel or 1 channel color to fill the mask with, depending on the input image
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])  # in case, the input image has a channel dimension

    roi_img = cv2.bitwise_and(edges, mask)

    # find the lines
    lines = cv2.HoughLinesP(roi_img, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

    final = draw_lines(img, lines)

    # draw the lines on the original image
    plt.subplot(2,2,3).axis("off")
    plt.imshow(final)

    plt.show()


if __name__ == '__main__':
    main()
