import cv2 as cv
from numpy import ones, uint8


def main() -> None:
    img = cv.imread("Photos/sudokidoki.jpg", 0)

    sobIMG = cv.Sobel(img, cv.CV_16S, dx=0, dy=1, ksize=5)
    lapIMG = cv.Laplacian(img, cv.CV_16S, ksize=5)

    cv.imshow("DIO", img)
    cv.waitKey(0)

    cv.imshow("DIO", sobIMG)
    cv.waitKey(0)

    cv.imshow("DIO", lapIMG)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
