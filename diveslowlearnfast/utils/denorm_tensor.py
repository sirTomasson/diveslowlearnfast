

def tensor_denorm(tensor, mean, std):
    if tensor.ndim == 5:
        return tensor.permute(0, 2, 3, 4, 1) * std[None, None, None, None, :] + mean[None, None, None, None, :]

    return tensor.permute(1, 2, 3, 0) * std[None, None, None, :] + mean[None, None, None, :]