import cv2 as cv
import openFuncs as of
import numpy as np


def main():
    imgOrj: np.ndarray = cv.imread("Photos/nekochanbeddo.jpg")
    img: np.ndarray = cv.imread("Photos/nekochanbeddo.jpg", 0)
    msk: np.ndarray = cv.imread("Photos/neko.jpg", 0)
    h, w = msk.shape
    green = (0, 255, 0)

    methods = [
        cv.TM_CCOEFF,
        # cv.TM_CCOEFF_NORMED,
        # cv.TM_CCORR,
        # cv.TM_CCORR_NORMED,
        # cv.TM_SQDIFF,
        # cv.TM_SQDIFF_NORMED,
    ]

    for method in methods:
        res = cv.matchTemplate(img, msk, method)

        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res)

        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            topLeft = minLoc
        else:
            topLeft = maxLoc

        bottomRight = (topLeft[0] + w, topLeft[1] + h)

        cv.rectangle(imgOrj, topLeft, bottomRight, green, 2)

        of.imshow(res, "gray")
        of.imshow(img, "gray")

    theRes = cv.cvtColor(imgOrj, cv.COLOR_BGR2RGB)
    of.imshow(theRes)


if __name__ == "__main__":
    main()
