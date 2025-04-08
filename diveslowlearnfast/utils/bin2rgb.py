import cv2

import numpy as np

def bin2rgb(bin):
    """
    Converts a binary mask to a 3 channel RGB image/video for display purposes.
    """
    subject_mask_video = np.float32(bin)
    subject_mask_video = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in subject_mask_video]
    subject_mask_video = np.stack(subject_mask_video)
    return np.uint8(subject_mask_video * 255)