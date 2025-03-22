import torch

import numpy as np

from .min_max_norm import tensor_min_max_norm
from .denorm_tensor import tensor_denorm

def postprocess_video(video, mean, std):
    assert type(video) == torch.Tensor
    video = tensor_denorm(video, mean, std)
    video = tensor_min_max_norm(video)
    return np.uint8(video.detach().cpu().numpy() * 255)