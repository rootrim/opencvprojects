import cv2 as cv
from numpy import ones, uint8


def main() -> None:
    img = cv.imread("Photos/sexyvamppire.png")
    linux = ones((5, 5), dtype=uint8)

    gradiatedIMG = cv.morphologyEx(img, cv.MORPH_GRADIENT, linux)

    cv.imshow("DIO", img)
    cv.waitKey(0)

    cv.imshow("DIO", gradiatedIMG)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
