import cv2 as cv
import pandas as pd
import time


def main():
    gt = pd.read_csv("Data/newgt.csv")

    vid = cv.VideoCapture("vid13sdp.mp4")

    tracker = cv.TrackerKCF_create()
    initBB = None
    fps = 25
    f = 0

    while True:
        start_frame_time = time.time()

        ret, frame = vid.read()
        if not ret:
            break

        frame = cv.resize(frame, dsize=(960, 540))
        (H, W) = frame.shape[:2]
        car_gt = gt[gt.frameNo.astype(int) == f]

        if len(car_gt) != 0:
            x, y, w, h = car_gt[["x", "y", "w", "h"]].iloc[0]
            centerX, centerY = car_gt[["centerX", "centerY"]].iloc[0]
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.circle(frame, (centerX, centerY), 2, (0, 0, 255), -1)

        if initBB is not None:
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(i) for i in box]
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv.imshow("Video", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord("t"):
            initBB = cv.selectROI("ROISelection", frame, fromCenter=False)
            if initBB != (0, 0, 0, 0):  # Geçerli bir ROI kontrolü
                tracker.init(frame, initBB)

        elif key == ord("q"):
            break

        f += 1

        # Döngü hızını sabitlemek için
        elapsed_time = time.time() - start_frame_time
        time.sleep(max(1 / fps - elapsed_time, 0))

    vid.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
