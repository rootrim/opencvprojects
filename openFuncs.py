import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def imshow(img: np.ndarray, cmap=""):
    plt.figure()
    if cmap != "":
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.show()


def saveVideo(
    path: str,
    vid: cv.VideoCapture,
    fps=0,
    frame_width=0,
    frame_height=0,
    codec="XVID",
    key="k",
) -> None:

    frameSpeed = fps or int(vid.get(cv.CAP_PROP_FPS))
    frame_width = frame_width or int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = frame_height or int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*codec)

    if not vid.isOpened():
        print("Video cannot be opened.")
        return

    output = cv.VideoWriter(
        path,
        fourcc,
        frameSpeed,
        (
            frame_width,
            frame_height,
        ),
    )

    while True:
        ret, frame = vid.read()
        cv.imshow("Frame", frame)

        if not ret or cv.waitKey(1) & 0xFF == ord(key):
            break

        output.write(frame)

    vid.release()
    output.release()
    print("Video saved.")


def imResize(img: np.ndarray, scale=1.0) -> np.ndarray:
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)

    dimension = (width, height)
    return cv.resize(img, dimension)


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
