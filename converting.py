import cv2 as cv


def main():
    img = cv.imread("Photos/cats.jpg")

    cvdImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow("Converted image", cvdImg)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
