import random

import numpy as np
import torch

import torch.nn.functional as F

from torchvision.transforms.v2 import Transform


class RandomRotateVideo(Transform):

    def __init__(self, max_angle, min_angle):
        super().__init__()
        self.max_angle = max_angle
        self.min_angle = min_angle

    def __call__(self, video):
        return rotate_video(video, self.random_angle())

    def random_angle(self):
        return random.uniform(self.max_angle, self.min_angle)


def rotate_video(video_tensor, angle_degrees):
    """
    Rotate a video tensor by a specified angle.

    Args:
        video_tensor: Tensor of shape (T, C, H, W)
        angle_degrees: Rotation angle in degrees

    Returns:
        Rotated video tensor of shape (T, C, H', W')
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_degrees)

    # Get tensor dimensions
    C, T, H, W = video_tensor.shape

    # Calculate output dimensions
    cos_theta_abs = abs(np.cos(angle_rad))
    sin_theta_abs = abs(np.sin(angle_rad))
    new_H = int(H * cos_theta_abs + W * sin_theta_abs)
    new_W = int(W * cos_theta_abs + H * sin_theta_abs)

    # Get actual sin and cos values for rotation
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    scale_h = 2.0 / (new_H - 1)
    scale_w = 2.0 / (new_W - 1)
    scale_orig_h = 2.0 / (H - 1)
    scale_orig_w = 2.0 / (W - 1)

    # Adjust transformation matrix for PyTorch's expected format
    transformation_matrix = torch.tensor([[
        [scale_w / scale_orig_w * cos_theta, scale_w / scale_orig_h * -sin_theta, 0],
        [scale_h / scale_orig_w * sin_theta, scale_h / scale_orig_h * cos_theta, 0]
    ]], dtype=video_tensor.dtype, device=video_tensor.device)

    # Repeat transformation matrix for each frame
    transformation_batch = transformation_matrix.repeat(C, 1, 1)

    # Create sampling grid
    grid = F.affine_grid(transformation_batch,
                         size=[C, T, H, W],
                         align_corners=True)

    # Apply rotation using grid sampling
    rotated_video = F.grid_sample(video_tensor,
                                  grid,
                                  mode='bilinear',
                                  padding_mode='zeros',
                                  align_corners=True)

    return rotated_video
