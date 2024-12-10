import cv2 as cv
import numpy as np


def main():
    kernel_size = (11, 11)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (11, 11))

    lower = (90, 100, 50)  # Alt sınır
    upper = (140, 255, 255)  # Üst sınır

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    while True:
        isopened, frame = cap.read()

        if isopened:
            blurFrame = cv.GaussianBlur(frame, kernel_size, 0)

            hsv = cv.cvtColor(blurFrame, cv.COLOR_BGR2HSV)

            mask = cv.inRange(hsv, lower, upper)

            mask_eroded = cv.erode(mask, None, iterations=2)
            mask_dilated = cv.dilate(mask_eroded, None, iterations=2)
            cv.imshow("Mask (Erosion & Dilation)", mask_dilated)

            (contours, _) = cv.findContours(
                mask.copy(),
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE,
            )

            # center = None

            if len(contours) > 0:

                c = max(contours, key=cv.contourArea)

                rect = cv.minAreaRect(c)

                ((x, y), (width, height), rotation) = rect

                s = "X: {}, Y: {}, Width: {}, Height: {}, Rotation: {}".format(
                    np.round(x),
                    np.round(y),
                    np.round(width),
                    np.round(height),
                    np.round(rotation),
                )
                print(s)

                theBox = cv.boxPoints(rect)
                theBox = np.int64(theBox)

                # M = cv.moments(c)
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                cv.drawContours(
                    frame,
                    [theBox],
                    0,
                    (0, 255, 0),
                    2,
                )

            cv.imshow("The Frame", frame)

        else:
            print("Kamera verisi alınamadı!")
            break

        if cv.waitKey(2) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
