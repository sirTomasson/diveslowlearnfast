import os
import cv2

import numpy as np

def flow2rgb(flow):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create HSV image where hue is direction and value is magnitude
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue (direction)
    hsv[..., 1] = 255                      # Saturation (full)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value (magnitude)

    # Convert HSV to RGB for visualization
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def read_flow(video_meta, video_dir, f='rgb'):
    assert f in ['rgb', '2channel']
    frames = []
    for idx in range(video_meta['start_frame'] + 1, video_meta['end_frame'] + 1):
        result = []
        for direction in ['x', 'y']:
            flow = cv2.imread(os.path.join(video_dir, 'flow',  video_meta['vid_name'], f'flow_{direction}_{idx:05d}.jpg'), cv2.IMREAD_GRAYSCALE)
            result.append((flow.astype(np.float32) - 128) / 128)

        frame = np.stack(result, axis=2)
        if f == 'rgb':
            frame = flow2rgb(frame)

        frames.append(frame)

    frames = np.stack(frames)

    return frames
