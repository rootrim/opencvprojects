import cv2 as cv


def main():
    kaskad = cv.CascadeClassifier("Data/haarcascade_frontalface_default.xml")

    cam = cv.VideoCapture(0)
    ret, frame = cam.read()

    if not ret:
        return

    faceKO = kaskad.detectMultiScale(frame)

    (x, y, w, h) = tuple(faceKO[0])
    track = (x, y, w, h)

    # NOTE:
    manofinterest = frame[y : y + h, x : x + w]
    haisenberginterest = cv.cvtColor(manofinterest, cv.COLOR_BGR2HSV)

    thesis = cv.calcHist([haisenberginterest], [0], None, [180], [0, 180])
    cv.normalize(haisenberginterest, haisenberginterest, 0, 255, cv.NORM_MINMAX)

    criticalDamange = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 5, 1)

    while True:
        ret, frame = cam.read()

        if not ret:
            break

        haisenburger = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        destination = cv.calcBackProject([haisenburger], [0], thesis, [0, 180], 1)

        ret, track = cv.meanShift(destination, track, criticalDamange)

        x, y, w, h = track

        zamuc = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

        cv.imshow("Track", zamuc)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
