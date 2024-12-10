import cv2 as cv
import numpy as np
import openFuncs as of


def main():
    img = cv.imread("Photos/symbols.jpg", 0)
    contours, hierarch = cv.findContours(
        img,
        cv.RETR_CCOMP,
        cv.CHAIN_APPROX_SIMPLE,
    )

    interCont = np.zeros(img.shape)
    exterCont = np.zeros(img.shape)

    print(hierarch)

    for i in range(len(contours)):
        # External kontürlerin hiyerarşideki konumu:
        if hierarch[0][i][3] == -1:
            cv.drawContours(exterCont, contours, i, 255, cv.FILLED)
        # External olmayan kontürler internaldir
        else:
            cv.drawContours(interCont, contours, i, 255, cv.FILLED)

    of.imshow(exterCont, "gray")
    of.imshow(interCont, "gray")


if __name__ == "__main__":
    main()
