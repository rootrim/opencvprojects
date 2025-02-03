import random

import cv2 as cv


def main():
    image = cv.imread("Photos/park.jpg")
    image = cv.resize(image, (600, 600))

    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)

    ss.switchToSelectiveSearchQuality()

    print("start")
    rects = ss.process()

    output = image.copy()

    for x, y, w, h in rects:
        color = [random.randint(0, 255) for j in range(0, 3)]
        cv.rectangle(output, (x, y), (x + w, y + h), color, 2)

    cv.imshow("Result", output)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
