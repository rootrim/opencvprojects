import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def main():
    baseImg = cv.imread("Photos/sudokidoki.jpg", 0)
    floImg = np.float32(baseImg)

    corners = cv.goodFeaturesToTrack(floImg, 740, 0.01, 3)
    corners = np.int64(corners)

    imshow(baseImg, "gray")

    for corner in corners:
        x, y = corner.ravel()
        cv.circle(baseImg, (x, y), 3, (125, 125, 125), -1)

    imshow(baseImg, "gray")


def imshow(img: np.ndarray, cmap: str):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
