import cv2 as cv
import openFuncs as of


def main():
    hog = cv.HOGDescriptor()

    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    img = cv.imread("Photos/group 1.jpg")

    # (rects, weights) = hog.detectMultiScale(img, padding=(8, 8), scale=1.05)
    (rects, weights) = hog.detectMultiScale(img)

    for x, y, w, h in rects:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    of.imshow(imgRGB)


if __name__ == "__main__":
    main()
