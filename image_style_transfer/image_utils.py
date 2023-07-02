import requests
import cv2
import numpy as np
from pathlib import Path


def load_image(url_or_path=""):
    url_or_path = str(url_or_path)

    if "http" in url_or_path:
        img_arr = np.asarray(
            bytearray(requests.get(url_or_path).content), dtype="uint8"
        )
        img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(url_or_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def downsample(img):
    return cv2.pyrDown(img)


def save_image(img, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        filename=str(path), img=img[:, :, :: -1], params=[cv2.IMWRITE_JPEG_QUALITY, 100]
    )
