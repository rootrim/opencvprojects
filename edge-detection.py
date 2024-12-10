import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def main():
    imgNotSigma = cv.imread("Photos/bigben.jpg", 0)

    blurredImg = cv.GaussianBlur(imgNotSigma, (3, 3), 7)

    median = np.median(blurredImg)

    low = int(max(0, (1 - 0.33) * median))
    high = int(min(255, (1 + 0.33) * median))

    edgedImg = cv.Canny(blurredImg, low, high)

    imshow(imgNotSigma, cmap="gray")

    imshow(edgedImg, cmap="gray")


def imshow(img: np.ndarray, cmap: str):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
