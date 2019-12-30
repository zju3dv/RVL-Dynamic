"""
Copyright SenseTime.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
import cv2

def read_grayscale_image(fn):
    img = cv2.imread(fn).astype(float) / 255.0
    shape = img.shape
    if len(shape) == 2:
        img = img[:, :, None]
    elif shape[2] > 1:
        img = img.mean(axis=2, keepdims=True)
    # img.shape = (w, h, 1)

    return img
