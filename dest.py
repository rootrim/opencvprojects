import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def display_image(title: str, img: np.ndarray, wait: int = 0) -> None:
    """Görseli bir pencerede görüntüle."""
    cv.imshow(title, img)
    cv.waitKey(wait)


def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Resmi belirtilen ölçeğe göre yeniden boyutlandır."""
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)
    return cv.resize(img, (new_width, new_height))


def add_text_to_image(
    img: np.ndarray, text: str, position: tuple, font_size: int = 1
) -> np.ndarray:
    """Resme metin ekle."""
    return cv.putText(
        img.copy(),  # Orijinal resmi bozmamak için kopya alınır
        text,
        position,
        cv.FONT_HERSHEY_COMPLEX,
        font_size,
        (255, 255, 255),
        2,
    )


def plot_histogram(img: np.ndarray) -> None:
    """Resmin histogramını matplotlib ile göster."""
    his_img = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(his_img)
    plt.xlim([0, 256])
    plt.show()


def main() -> None:
    img: np.ndarray = cv.imread("Photos/park.jpg", 0)

    # Orijinal resmi göster
    display_image("Original Image", img)

    print("Original Shape:", img.shape)

    # Resmi yeniden boyutlandır
    res_img = resize_image(img, scale=0.8)
    display_image("Resized Image", res_img)
    print("Resized Shape:", res_img.shape)

    # Resme metin ekle
    text_img = add_text_to_image(
        img, "KONODIDODA", (img.shape[1] // 4, img.shape[0] // 2), font_size=1
    )
    display_image("Image with Text", text_img)

    # Threshold işlemi
    _, thr_img = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
    display_image("Threshold Image", thr_img)

    # Gaussian Blur işlemi
    gas_img = cv.GaussianBlur(img, (3, 3), 7)
    display_image("Gaussian Blurred Image", gas_img)

    # Laplacian kenar algılama
    lap_img = cv.Laplacian(gas_img, cv.CV_64F, ksize=5)
    display_image(
        "Laplacian Image", cv.convertScaleAbs(lap_img)
    )  # Görüntüyü uint8'e dönüştür

    # Histogramı matplotlib ile göster
    plot_histogram(img)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
