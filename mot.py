import cv2 as cv


def main():
    vid = cv.VideoCapture("vid13sdp.mp4")
    trackers: cv.legacy_MultiTracker = cv.legacy.MultiTracker_create()

    while True:
        ret, frame = vid.read()
        frame = cv.resize(frame, dsize=(960, 540))

        if ret:
            success, boxes = trackers.update(frame)

            if success:
                for box in boxes:
                    (x, y, w, h) = [int(i) for i in box]
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow("Video", frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord("t"):

            tracker = cv.legacy.TrackerKCF_create()
            box = cv.selectROI("ROISelection", frame, fromCenter=False)
            if box != (0, 0, 0, 0):
                trackers.add(tracker, frame, box)

        elif key == ord("q"):
            break

    vid.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
