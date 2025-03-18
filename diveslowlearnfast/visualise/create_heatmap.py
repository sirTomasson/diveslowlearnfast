import torch

import matplotlib.pyplot as plt


def revert_tensor_normalize(tensor, mean, std):
    """
    Revert normalization for a given tensor by multiplying by the std and adding the mean.
    Args:
        tensor (tensor): tensor to revert normalization.
        mean (tensor or list): mean value to add.
        std (tensor or list): std to multiply.
    """
    if type(mean) == list or type(mean) == tuple:
        mean = torch.tensor(mean)
    if type(std) == list or type(std) == tuple:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor


def create_heatmaps(inputs, localisation_maps, mean, std, alpha=0.5, colormap='viridis'):
    colormap = plt.get_cmap(colormap)
    results = []
    for i, localization_map in enumerate(localisation_maps):
        # Convert (B, 1, T, H, W) to (B, T, H, W)
        localization_map = localization_map.squeeze(dim=1)
        if localization_map.device != torch.device("cpu"):
            localization_map = localization_map.cpu()
        heatmap = colormap(localization_map)
        heatmap = heatmap[:, :, :, :, :3]
        # Permute input from (B, C, T, H, W) to (B, T, H, W, C)
        curr_inp = inputs[i].permute(0, 2, 3, 4, 1)
        if curr_inp.device != torch.device("cpu"):
            curr_inp = curr_inp.cpu()
        curr_inp = revert_tensor_normalize(
            curr_inp, mean, std
        )
        heatmap = torch.from_numpy(heatmap)
        curr_inp = alpha * heatmap + (1 - alpha) * curr_inp
        # Permute inp to (B, C, T, H, W)
        curr_inp = curr_inp.permute(0, 4, 1, 2, 3)
        results.append(curr_inp)

    return results
