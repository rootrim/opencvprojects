import cv2 as cv
import openFuncs as of


def main(dio: bool) -> None:
    chs = cv.imread("Photos/cukulatalar.jpg", 0)
    cho = cv.imread("Photos/cukulata.jpg", 0)

    if dio:
        display_imgs(chs, cho, cmap="gray")

    orb: cv.ORB = cv.ORB_create()

    kp1, ds1 = orb.detectAndCompute(cho, None)
    kp2, ds2 = orb.detectAndCompute(chs, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING)

    matches = bf.match(ds1, ds2)
    smatchs = sorted(matches, key=lambda x: x.distance)

    imgMatch = cv.drawMatches(cho, kp1, chs, kp2, matches[:30], None, flags=2)
    of.imshow(imgMatch, "gray")


def display_imgs(*imgs, cmap="gray"):
    for img in imgs:
        of.imshow(img, cmap=cmap)


if __name__ == "__main__":
    main(False)
