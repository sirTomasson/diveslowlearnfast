import cv2

import numpy as np

def get_sample(dataset):
    result = next(iter(dataset))
    return result[0], result[1]


