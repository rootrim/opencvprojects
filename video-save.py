import cv2 as cv


def main():
    myVideo = cv.VideoCapture(0)
    saveVideo("Videos/kamera.mp4", myVideo, codec="mp4v")


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


if __name__ == "__main__":
    main()
