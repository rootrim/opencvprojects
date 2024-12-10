import cv2 as cv
import openFuncs as of


def main():
    orjNeko = cv.imread("Photos/nekochanbeddo.jpg")
    neko = cv.imread("Photos/nekochanbeddo.jpg", 0)
    of.imshow(neko, "gray")

    neko[760:900, 480:640]

    cv.imwrite("Photos/neko.jpg", orjNeko[760:900, 480:640])


if __name__ == "__main__":
    main()
