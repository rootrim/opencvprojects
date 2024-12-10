import cv2 as cv
import numpy as np


def main():
    im = cv.imread("Photos/cat.jpg")

    imHStacked = np.hstack((im, im))
    imVStacked = np.vstack((im, im))

    cv.imshow("Horizontly stacked image", imHStacked)

    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imshow("Verticly stacked image", imVStacked)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
