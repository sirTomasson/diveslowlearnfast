import os
import cv2

import numpy as np

def _video_flow2rgb(flow):
    return np.stack([flow2rgb(f) for f in flow])

def flow2rgb(flow, mode='frame'):
    assert mode in ['frame', 'video', 'batch']

    if mode == 'video':
        return _video_flow2rgb(flow)

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create HSV image where hue is direction and value is magnitude
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue (direction)
    hsv[..., 1] = 255                      # Saturation (full)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value (magnitude)

    # Convert HSV to RGB for visualization
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def rgb2flow(rgb_image):
    # Convert RGB to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Extract components
    h = hsv[..., 0] * np.pi / 90.0  # Convert hue to angle (reverse of angle * 180 / np.pi / 2)
    s = hsv[..., 1]  # Saturation (not used in conversion)
    v = hsv[..., 2]  # Value represents magnitude

    # Create flow matrix with same shape as input but with 2 channels
    flow = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 2), dtype=np.float32)

    # Convert polar coordinates (angle and magnitude) back to cartesian (x,y)
    magnitude = v  # Assuming magnitude was normalized to 0-255 range
    flow[..., 0] = magnitude * np.cos(h)  # x component
    flow[..., 1] = magnitude * np.sin(h)  # y component

    return flow

def read_flow(video_meta, video_dir, modality='rgb'):
    assert modality in ['rgb', 'flow']
    frames = []
    for idx in range(video_meta['start_frame'] + 1, video_meta['end_frame'] + 1):
        result = []
        for direction in ['x', 'y']:
            flow = cv2.imread(os.path.join(video_dir, 'flow',  video_meta['vid_name'], f'flow_{direction}_{idx:05d}.jpg'), cv2.IMREAD_GRAYSCALE)
            result.append((flow.astype(np.float32) - 128) / 128)

        frame = np.stack(result, axis=2)
        if modality == 'rgb':
            frame = flow2rgb(frame)

        frames.append(frame)

    frames = np.stack(frames)

    return frames


def flow(frames):
    frame1 = frames[0]
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

    frame_idx = 0
    out_frames = []

    for frame2 in frames[1:]:
        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = tvl1.calc(prev_gray, gray, None)

        out_frames.append(flow)

        prev_gray = gray
        frame_idx += 1

    return np.stack(out_frames)

