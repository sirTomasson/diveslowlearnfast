import math

import torch


def superimpose_confounder(
        x: torch.Tensor,
        y,
        size=24,
        grid_size=48,
        channel=0,
        inplace=False,
):
    """

    Args:
        x (torch.Tensor): the input video tensor in C x T x H x W
        y (torch.Tensor): the target video tensor
        size (int): the size of the confounder
        grid_size (int): the number of confounders on the grid, this should uniquely map classes in the dataset
        channel (int): the color channel which should set to 1, a number between 0-2
        inplace (bool): whether to perform inplace operations
    """
    assert channel <= 2

    if not inplace:
        x = x.clone()

    if type(y) == torch.Tensor:
        y = y.item()

    _, _, H, W = x.shape

    g = math.sqrt(grid_size)

    gh = int(H / g)
    gw = int(W / g)

    r = int(y / g)
    c = int(y % g)
    tr = r * gh
    tc = c * gw

    # modify the `channel` at denormalised row and column
    # and set all other channels to zero
    color_mask = torch.tensor([0, 0, 0])
    color_mask[channel] = 1
    x[:, :, tr:tr+size, tc:tc+size] = color_mask[:, None, None, None]
    return x

