import pandas as pd
import cv2 as cv
import numpy as np


def main():
    videopath = "vid13sdp.mp4"
    cap = cv.VideoCapture(videopath)
    fps = 25
    delay = int(1000 / fps)
    cols = [
        "frame_number",
        "identity_number",
        "left",
        "top",
        "width",
        "height",
        "score",
        "class",
        "visibility",
    ]
    data = pd.read_csv("Videos/MOT17/train/MOT17-13-SDP/gt/gt.txt", names=cols)

    car = data[data["class"] == 3]
    cardi = 29
    maxframe = np.max(data["frame_number"])
    boundingBoxes = []

    for i in range(1, maxframe - 1):
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv.resize(frame, dsize=(960, 540))

        cardifter = np.logical_and(
            car["frame_number"] == i,
            car["identity_number"] == cardi,
        )

        if cardifter.any():
            x = int(car[cardifter].left.values[0] / 2)
            y = int(car[cardifter].top.values[0] / 2)
            w = int(car[cardifter].width.values[0] / 2)
            h = int(car[cardifter].height.values[0] / 2)

            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (0, 0, 255), -1)

        boundingBoxes.append((i, x, y, w, h, int(x + w / 2), int(y + h / 2)))

        cv.imshow("Frame", frame)
        if cv.waitKey(delay) & 0xFF == ord("q"):
            break

    df = pd.DataFrame(
        boundingBoxes,
        columns=[
            "frameNo",
            "x",
            "y",
            "w",
            "h",
            "centerX",
            "centerY",
        ],
    )
    df.to_csv("Data/newgt.csv", index=False)
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
