from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImgUtil:
    @staticmethod
    def read_img(path: str) -> np.ndarray:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def show_img(img: Union[str, np.ndarray]):
        if isinstance(img, str):
            img = ImgUtil.read_img(img)
        plt.imshow(img)
        plt.show()
