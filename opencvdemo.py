import cv2 as cv


def main():
    vid = cv.VideoCapture("Videos/kitten.mp4")
    vishow(vid, 30, "c")


def vishow(vid: cv.VideoCapture, ms: int, key: str):
    while True:
        ret, frame = vid.read()

        if not ret:
            break

        cv.imshow("Frame", frame)

        if cv.waitKey(ms) & 0xFF == ord(key):
            break

    vid.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
