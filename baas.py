import cv2 as cv
import openFuncs as of
import numpy as np


def main():
    img: np.ndarray = cv.imread("Photos/dots.png")

    blur: np.ndarray = cv.medianBlur(img, 13)

    gray: np.ndarray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

    _, threshed = cv.threshold(gray, 65, 255, cv.THRESH_BINARY)

    of.imshow(threshed, "gray")

    dist = cv.distanceTransform(threshed, cv.DIST_L2, 5)

    of.imshow(dist, "gray")

    _, foreg = cv.threshold(dist, 0, 255, 0)

    linux = np.ones((3, 3), np.uint8)

    backg = cv.dilate(foreg, linux, iterations=1)
    foreg = np.uint8(foreg)

    known = cv.subtract(backg, foreg, dtype=0)

    _, marker = cv.connectedComponents(foreg)
    marker += 1
    marker[known == 255] = 0

    marker = cv.watershed(img, marker)

    contours, hierarchy = cv.findContours(
        marker.copy(),
        cv.RETR_CCOMP,
        cv.CHAIN_APPROX_SIMPLE,
    )

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv.drawContours(img, contours, i, (0, 255, 0), 2)

    of.imshow(img)


if __name__ == "__main__":
    main()
