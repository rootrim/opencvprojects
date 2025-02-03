import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from typing import Any, Generator


def image_pyramid(
    image: NDArray[Any], scale: float = 1.5, minSize: tuple[int, int] = (224, 224)
) -> Generator[NDArray[Any], None, None]:
    yield image

    while True:
        h = int(image.shape[0] / scale)
        w = int(image.shape[1] / scale)

        if h < minSize[0] or w < minSize[1]:
            break

        image = cv.resize(image, (w, h))
        yield image
