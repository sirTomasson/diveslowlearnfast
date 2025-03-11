import torch

def tensor_denorm(tensor, mean, std, permute_channel=True):
    if type(mean) != torch.Tensor:
        mean = torch.tensor(mean)

    if type(std) != torch.Tensor:
        std = torch.tensor(std)

    if tensor.ndim == 5:
        if permute_channel:
            return tensor.permute(0, 2, 3, 4, 1) * std[None, None, None, None, :] + mean[None, None, None, None, :]

        return tensor * std[:, None, None, None, None] + mean[:, None, None, None, None]

    if permute_channel:
        return tensor.permute(1, 2, 3, 0) * std[None, None, None, :] + mean[None, None, None, :]

    return tensor * std[:, None, None, None] + mean[:, None, None, None]