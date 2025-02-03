import cv2 as cv
import numpy as np


def main():
    kernel_size = (11, 11)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)

    lower = (90, 100, 50)  # Alt sınır (Mavi)
    upper = (140, 255, 255)  # Üst sınır (Mavi)

    cap = cv.VideoCapture(0)

    # GPU için Gaussian Blur filtresi oluştur (CUDA hızlandırmalı)
    gaussian_filter = cv.cuda.createGaussianFilter(
        cv.CV_8UC3, cv.CV_8UC3, kernel_size, 0
    )

    # GPU için Erosion ve Dilation işlemleri oluştur
    erode_filter = cv.cuda.createMorphologyFilter(cv.MORPH_ERODE, cv.CV_8UC1, kernel)
    dilate_filter = cv.cuda.createMorphologyFilter(cv.MORPH_DILATE, cv.CV_8UC1, kernel)

    while True:
        isopened, frame = cap.read()
        if not isopened:
            print("Kamera verisi alınamadı!")
            break

        # Görüntüyü GPU belleğine yükle
        gpu_frame = cv.cuda_GpuMat()
        gpu_frame.upload(frame)

        # Gaussian Blur (GPU)
        gpu_blur = gaussian_filter.apply(gpu_frame)

        # BGR'den HSV'ye dönüşüm (GPU)
        gpu_hsv = cv.cuda.cvtColor(gpu_blur, cv.COLOR_BGR2HSV)

        # Renk maskesi oluşturma (GPU)
        gpu_mask = cv.cuda.inRange(gpu_hsv, lower, upper)

        # GPU'da erosion işlemi
        gpu_mask_eroded = erode_filter.apply(gpu_mask)

        # GPU'da dilation işlemi
        gpu_mask_dilated = dilate_filter.apply(gpu_mask_eroded)

        # Maskeyi CPU'ya çekip göster (Çünkü cv.findContours GPU'da yok)
        mask_dilated = gpu_mask_dilated.download()
        cv.imshow("Mask (Erosion & Dilation)", mask_dilated)

        # Kontur tespiti (CPU'da çalışıyor)
        contours, _ = cv.findContours(
            mask_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            c = max(contours, key=cv.contourArea)
            rect = cv.minAreaRect(c)
            theBox = cv.boxPoints(rect)
            theBox = np.int64(theBox)

            # Çerçeveyi çiz
            cv.drawContours(frame, [theBox], 0, (0, 255, 0), 2)

        # Sonucu ekrana bas
        cv.imshow("The Frame", frame)

        if cv.waitKey(2) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
