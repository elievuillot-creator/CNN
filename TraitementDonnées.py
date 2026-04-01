import numpy as np  # linear algebra
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt


def image_to_pixel_matrix(filepath):
    img = plt.imread(filepath)

    # RGBA → RGB si besoin
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # float [0,1] → int [0,255] si besoin
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # On force explicitement la forme (H, W, 3)
    img = img.reshape(img.shape[0], img.shape[1], 3)

    return img
