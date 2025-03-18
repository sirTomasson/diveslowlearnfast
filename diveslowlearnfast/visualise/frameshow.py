
import torch

import numpy as np
import matplotlib.pyplot as plt

from typing import Union



def frameshow(tensor_or_ndarray: Union[torch.Tensor, np.ndarray], frame_idx=0, axis='off', title=None, permute=None):
    if permute is None:
        permute = [1, 2, 3, 0]

    if type(tensor_or_ndarray) == torch.Tensor:
        tensor_or_ndarray = tensor_or_ndarray.detach().cpu().permute(*permute).numpy()

    plt.imshow(tensor_or_ndarray[frame_idx])
    if title is not None:
        plt.title(title)
    plt.axis(axis)
    plt.show()