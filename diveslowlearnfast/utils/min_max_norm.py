
def tensor_min_max_norm(video_tensor):
    """

    Args:
        video_tensor (Tensor): Tensor of shape (C, T, H, W)
    """
    assert video_tensor.ndim == 4

    return (video_tensor - video_tensor.min()) / (video_tensor.max() - video_tensor.min())