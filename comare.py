import cv2 as cv


def main() -> None:
    cam = cv.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        if not ret:
            break

    cv.imshow("Frame", frame)
    cv.waitKey(0)
    cv.imwrite("Cam/fresh.jpg", frame)


if __name__ == "__main__":
    main()
