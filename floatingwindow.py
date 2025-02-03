import cv2 as cv


def main():
    image = cv.imread("Photos/bigben.jpg")
    slider = sliding_window(image, 5, (200, 150))
    for i, (_, _, img) in enumerate(slider):
        print(i)
        if i % 100 == 0:
            cv.imshow("Amazing Photo", img)
            cv.waitKey(0)


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0] - windowSize[1], stepSize):
        for x in range(0, image.shape[1] - windowSize[0], stepSize):
            yield (x, y, image[y : y + windowSize[1], x : x + windowSize[0]])


if __name__ == "__main__":
    main()
