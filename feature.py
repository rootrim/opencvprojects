import cv2 as cv
import openFuncs as of


def main(dio: bool, kitty: bool) -> None:
    if kitty:
        of.kitty()

    thepicture = cv.imread("Photos/cukulatalar.jpg", 0)
    chocolata = cv.imread("Photos/cukulata.jpg", 0)

    if dio:
        display_imgs(thepicture, chocolata, cmap="gray")

    orb: cv.ORB = cv.ORB_create()

    kp1, ds1 = orb.detectAndCompute(chocolata, None)
    kp2, ds2 = orb.detectAndCompute(thepicture, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING)

    matches = bf.match(ds1, ds2)
    smatchs = sorted(matches, key=lambda x: x.distance)

    imgMatch = cv.drawMatches(
        chocolata,
        kp1,
        thepicture,
        kp2,
        smatchs[:20],
        None,
        flags=2,
    )

    of.imshow(imgMatch, "gray")

    sift: cv.SIFT = cv.SIFT_create()

    bf = cv.BFMatcher()

    kp1, ds1 = sift.detectAndCompute(chocolata, None)
    kp2, ds2 = sift.detectAndCompute(thepicture, None)

    matches = bf.knnMatch(ds1, ds2, k=2)

    nices = [[m for m, n in matches if m.distance < 0.75 * n.distance]]

    siftMatches = cv.drawMatchesKnn(
        chocolata,
        kp1,
        thepicture,
        kp2,
        nices,
        None,
        flags=2,
    )

    of.imshow(siftMatches)
    cv.destroyAllWindows()


def display_imgs(*imgs, cmap="gray"):
    for img in imgs:
        of.imshow(img, cmap=cmap)


if __name__ == "__main__":
    main(False, False)
