import cv2 as cv
import os
from os.path import isfile, join


def main() -> None:
    pathin = "Videos/MOT17/train/MOT17-13-SDP/img1"
    pathout = "vid13sdp.mp4"
    fps = 25
    delay = int(1000 / fps)
    resulation = (1920, 1080)
    out = cv.VideoWriter(
        pathout,
        cv.VideoWriter_fourcc(*"MP4V"),
        fps,
        resulation,
        True,
    )

    imgs = [
        # cv.imread(join(pathin, f))
        join(pathin, f)
        for f in os.listdir(pathin)
        if isfile(join(pathin, f))
    ]
    sortedimgs = sorted(imgs, key=lambda x: x[-7:-3])

    for path in sortedimgs:
        img = cv.imread(path)
        print(f"{path} was readen")
        cv.imshow("Video", img)
        out.write(img)
        print(f"{path} was writed")

        key = cv.waitKey(delay)
        if key == ord("q"):
            break

    out.release()


if __name__ == "__main__":
    main()
