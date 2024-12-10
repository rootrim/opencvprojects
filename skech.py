import cv2 as cv
from numpy import zeros, ndarray


def main() -> None:
    blank = zeros((500, 500, 3), dtype="uint8")
    imshow(blank)

    blankMidW = blank.shape[1] // 2
    blankMidH = blank.shape[0] // 2

    green = (0, 255, 0)

    cv.rectangle(
        blank,
        (blankMidW - 50, blankMidH - 50),
        (blankMidW + 50, blankMidH + 50),
        green,
        thickness=2,
    )
    imshow(blank)

    cv.line(
        blank,
        (blankMidW - 50, blankMidH - 50),
        (blankMidW + 50, blankMidH + 50),
        green,
        thickness=2,
    )
    imshow(blank)

    cv.circle(blank, (blankMidW, blankMidH), 40, green, thickness=2)
    imshow(blank)

    cv.putText(
        blank,
        "Konnichiwa,",
        (blankMidW - 50, blankMidH + 70),
        cv.FONT_HERSHEY_TRIPLEX,
        0.75,
        green,
    )
    imshow(blank)

    cv.putText(
        blank,
        "Za Warudo!",
        (blankMidW - 50, blankMidH + 90),
        cv.FONT_HERSHEY_COMPLEX,
        0.75,
        green,
    )
    imshow(blank)


def imshow(img: ndarray, name="Image", ms=0) -> None:
    cv.imshow(name, img)
    cv.waitKey(ms)


if __name__ == "__main__":
    main()
