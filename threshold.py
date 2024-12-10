import cv2 as cv
import matplotlib.pyplot as plt


def main():
    img = cv.imread("Photos/cat.jpg")
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    imgThreshed = cv.adaptiveThreshold(
        imgGray,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY,
        11,
        8,
    )

    cv.imshow("Final Result", imgThreshed)
    cv.waitKey(0)

    plt.imshow(imgThreshed, cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
