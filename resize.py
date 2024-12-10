import cv2 as cv
from numpy import ndarray


def main():
    img = cv.imread("Photos/cat.jpg")

    resImg = imResize(img, 0.01)

    cv.imshow("Photo", img)
    cv.waitKey(0)

    cv.imshow("Resized Photo", resImg)
    cv.waitKey(0)


def imResize(img: ndarray, scale=1.0) -> ndarray:
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    dimension = (width, height)
    return cv.resize(img, dimension)


if __name__ == "__main__":
    main()
