import cv2 as cv


def main():
    img = cv.imread("Photos/cat.jpg")

    normalBlur = cv.blur(img, (5, 5))
    gaussBlur = cv.GaussianBlur(img, (5, 5), 7)
    medianBlur = cv.medianBlur(img, 3)

    cv.imshow("Normal", normalBlur)
    cv.waitKey(0)

    cv.imshow("Gauss", gaussBlur)
    cv.waitKey(0)

    cv.imshow("Median", medianBlur)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
