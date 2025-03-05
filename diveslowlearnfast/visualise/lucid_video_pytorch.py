import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Union, Callable
import time

from tqdm.notebook import tqdm as tqdm_notebook
import matplotlib.pyplot as plt
from torchvision import transforms

from diveslowlearnfast.config import Config
from diveslowlearnfast.models import SlowFast, utils as model_utils



class SlowFastLayer:
    """Represents a layer in the SlowFast model with name, depth, and tags"""

    def __init__(self, name: str, depth: int, tags: List[str]):
        self.name = name
        self.depth = depth
        self.tags = tags

class SlowFastLucid(nn.Module):
    """PyTorch implementation of SlowFast model to extract activations"""

    def __init__(self, model: SlowFast):
        super(SlowFastLucid, self).__init__()
        self.model = model
        self.dataset = 'Diving48'
        self.image_shape = [None, None, None, 3]
        self.image_rank = 4
        self.image_value_range = (0, 1)
        self.input_name = 'input'

        # Define the layers
        self.layers = []
        self._define_layers()

        # Set hooks to capture activations
        self._register_hooks()

    def _define_layers(self):
        """Define the layers structure"""
        layer_defs = [
            # Stem layers
            {'tags': ['conv'], 'name': 'slow_stem', 'depth': 64},
            {'tags': ['conv'], 'name': 'fast_stem', 'depth': 8},

            # After first fusion
            {'tags': ['conv'], 'name': 's1_fuse_output_slow', 'depth': 80},  # 64 + 16
            {'tags': ['conv'], 'name': 's1_fuse_output_fast', 'depth': 8},

            # Stage 2 outputs
            {'tags': ['conv'], 'name': 's2_pathway0_output', 'depth': 256},
            {'tags': ['conv'], 'name': 's2_pathway1_output', 'depth': 32},

            # Stage 3 outputs
            {'tags': ['conv'], 'name': 's3_pathway0_output', 'depth': 512},
            {'tags': ['conv'], 'name': 's3_pathway1_output', 'depth': 64},

            # Stage 4 outputs
            {'tags': ['conv'], 'name': 's4_pathway0_output', 'depth': 1024},
            {'tags': ['conv'], 'name': 's4_pathway1_output', 'depth': 128},

            # Stage 5 outputs
            {'tags': ['conv'], 'name': 's5_pathway0_output', 'depth': 2048},
            {'tags': ['conv'], 'name': 's5_pathway1_output', 'depth': 256},

            # Pre-classification features
            {'tags': ['conv'], 'name': 'final_features', 'depth': 2304}  # 2048 + 256
        ]

        self.layers = [SlowFastLayer(**layer) for layer in layer_defs]

    def _register_hooks(self):
        """Register forward hooks to capture activations at key points in the SlowFast model"""
        self.activations = {}

        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output

            return hook

        # Register hooks for stem layers
        self.model.s1.pathway0_stem.register_forward_hook(get_activation('slow_stem'))
        self.model.s1.pathway1_stem.register_forward_hook(get_activation('fast_stem'))

        # Register hooks after fusions
        self.model.s1_fuse.register_forward_hook(get_activation('s1_fuse_output'))
        self.model.s2_fuse.register_forward_hook(get_activation('s2_fuse_output'))
        self.model.s3_fuse.register_forward_hook(get_activation('s3_fuse_output'))
        self.model.s4_fuse.register_forward_hook(get_activation('s4_fuse_output'))

        # Register hooks at the output of each stage for both pathways
        self.model.s2.pathway0_res2.register_forward_hook(get_activation('s2_pathway0_output'))
        self.model.s2.pathway1_res2.register_forward_hook(get_activation('s2_pathway1_output'))

        self.model.s3.pathway0_res3.register_forward_hook(get_activation('s3_pathway0_output'))
        self.model.s3.pathway1_res3.register_forward_hook(get_activation('s3_pathway1_output'))

        self.model.s4.pathway0_res5.register_forward_hook(get_activation('s4_pathway0_output'))
        self.model.s4.pathway1_res5.register_forward_hook(get_activation('s4_pathway1_output'))

        self.model.s5.pathway0_res2.register_forward_hook(get_activation('s5_pathway0_output'))
        self.model.s5.pathway1_res2.register_forward_hook(get_activation('s5_pathway1_output'))

    def layer_groups(self):
        """Return layers grouped into lower, middle, and upper layers"""
        return OrderedDict([
            ('lower_layers', self.layers[:3]),
            ('middle_layers', self.layers[3:8]),
            ('upper_layers', self.layers[9:-1])
        ])

    def forward(self, x):
        """Forward pass through the model"""
        # This would be the actual model implementation
        # For now, we'll use this as a placeholder
        return self.model(x)


#
# Visualization utils
#

def display_videos(videos_list, figsize=(12, 6)):
    """Display a list of videos"""
    # For simplicity, we'll just show the first frame of each video
    fig, axes = plt.subplots(1, len(videos_list), figsize=figsize)

    if len(videos_list) == 1:
        axes = [axes]

    for i, video in enumerate(videos_list):
        # Convert from PyTorch tensor to numpy
        if torch.is_tensor(video):
            video = video.detach().cpu().numpy()

        # Show first frame from first batch
        frame = video[0, 0]

        # If normalized to [0,1]
        if frame.max() <= 1.0:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

        if frame.shape[0] == 3:  # Handle channel-first format
            frame = np.transpose(frame, (1, 2, 0))

        axes[i].imshow(frame)
        axes[i].set_title(f"Video {i + 1}")
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def visstd(x, s=0.1):
    """Normalize the image range for visualization"""
    # Assuming input is a tensor
    if torch.is_tensor(x):
        mean = torch.mean(x)
        std = torch.std(x)
        return torch.clip(s * (x - mean) / std + 0.5, 0, 1)
    else:
        # Numpy case
        mean = np.mean(x)
        std = np.std(x)
        return np.clip(s * (x - mean) / std + 0.5, 0, 1)


#
# Objective functions
#

class ObjectiveFunction:
    """Base class for visualization objectives"""

    def __call__(self, model, input_video):
        raise NotImplementedError("Subclasses must implement this method")


class NeuronObjective(ObjectiveFunction):
    """Objective to visualize what activates a specific neuron"""

    def __init__(self, layer_name, channel_n, x=None, y=None, t=None, batch=None):
        self.layer_name = layer_name
        self.channel_n = channel_n
        self.x = x
        self.y = y
        self.t = t
        self.batch = batch

    def __call__(self, model, _input_video):
        # Get the activation
        activation = model.activations.get(self.layer_name)

        if activation is None:
            raise ValueError(f"Layer {self.layer_name} activation not found")

        # Determine indices
        batch_idx = 0 if self.batch is None else self.batch
        t_idx = activation.shape[2] // 2 if self.t is None else self.t
        x_idx = activation.shape[3] // 2 if self.x is None else self.x
        y_idx = activation.shape[4] // 2 if self.y is None else self.y

        # Get the neuron activation
        result = activation[batch_idx, self.channel_n, t_idx, x_idx, y_idx]
        return result


class FrameObjective(ObjectiveFunction):
    """Objective to visualize what activates a specific frame"""

    def __init__(self, layer_name, channel_n, t=None, batch=None):
        self.layer_name = layer_name
        self.channel_n = channel_n
        self.t = t
        self.batch = batch

    def __call__(self, model, input_video):
        # Get the activation
        activation = model.activations.get(self.layer_name)

        if activation is None:
            raise ValueError(f"Layer {self.layer_name} activation not found")

        # Determine indices
        batch_idx = 0 if self.batch is None else self.batch
        t_idx = activation.shape[2] // 2 if self.t is None else self.t

        # Get the frame activation
        result = activation[batch_idx, self.channel_n, t_idx, :, :]
        # Take the mean for a scalar value
        return torch.mean(result)


class AlignmentObjective(ObjectiveFunction):
    """Objective to encourage temporal consistency"""

    def __init__(self, layer_name, frames_n, decay_ratio=2):
        self.layer_name = layer_name
        self.frames_n = frames_n
        self.decay_ratio = decay_ratio

    def __call__(self, model, input_video):
        # Get the activation
        activation = model.activations.get(self.layer_name)

        if activation is None:
            raise ValueError(f"Layer {self.layer_name} activation not found")

        accum = 0
        for d in [1]:  # Original used [1] but mentioned [1, 2, 3, 4]
            for i in range(self.frames_n - d):
                a, b = i, i + d
                arr1, arr2 = activation[:, :, a], activation[:, :, b]
                accum += torch.mean((arr1 - arr2) ** 2) / self.decay_ratio ** float(d)

        # Negate because we want to minimize differences (maximize negative differences)
        return -accum


#
# Video parameterization
#

def fft_video(shape, sd=0.01, decay_power=1, device='cuda'):
    """
    Generate a video using frequency domain parameterization.

    This creates a random spectrum in the frequency domain and converts it
    to the spatial domain using inverse FFT.

    Args:
        shape: Tuple (batch, t, h, w, ch) specifying video dimensions
        sd: Standard deviation for random initialization
        decay_power: Power for frequency decay (higher values create smoother videos)
        device: Device to place tensor on ('cuda' or 'cpu')

    Returns:
        PyTorch tensor of shape (batch, t, h, w, ch)
    """
    batch, t, h, w, ch = shape

    # Initialize random spectrum parameters in frequency domain
    spectrum_shape = (batch, ch, t, h, w // 2 + 1, 2)  # Last dim: real/imaginary
    spectrum_params = torch.randn(spectrum_shape, device=device) * sd

    # Create frequency grid for decay calculation
    freqs_t = torch.fft.fftfreq(t)[None, None, :, None, None].to(device)
    freqs_y = torch.fft.fftfreq(h)[None, None, None, :, None].to(device)
    freqs_x = torch.fft.rfftfreq(w)[None, None, None, None, :].to(device)

    # Calculate frequency decay scale
    freqs = torch.sqrt(freqs_t ** 2 + freqs_y ** 2 + freqs_x ** 2)
    scale = 1.0 / torch.maximum(freqs, torch.tensor(1.0 / (max(w, h)), device=device)) ** decay_power
    scale = scale * torch.sqrt(torch.tensor(w * h, dtype=torch.float32, device=device))

    # Apply scaling to the spectrum parameters
    spectrum_real = spectrum_params[:, :, :, :, :, 0] * scale
    spectrum_imag = spectrum_params[:, :, :, :, :, 1] * scale

    # Create complex spectrum
    spectrum = torch.complex(spectrum_real, spectrum_imag)

    # Perform inverse FFT to get spatial representation
    videos = []
    for b in range(batch):
        video_batch = []
        for c in range(ch):
            # IFFT returns complex tensor, we take real part
            video_channel = torch.fft.irfftn(spectrum[b, c], (t, h, w)).real
            video_batch.append(video_channel)

        # Stack channels
        video = torch.stack(video_batch, dim=-1)
        videos.append(video)

    # Stack batches
    result = torch.stack(videos, dim=0)

    # Normalize to have reasonable range
    result = result / 4.0

    return result


def to_valid_rgb(video, decorrelate=True, sigmoid=True):
    """
    Convert a video tensor to valid RGB values

    Args:
        video: Input video tensor of shape (batch, t, h, w, 3)
        decorrelate: Whether to decorrelate colors
        sigmoid: Whether to apply sigmoid for bounding to [0, 1]

    Returns:
        RGB video tensor with values between 0 and 1
    """
    # Extract just the RGB channels
    if video.shape[-1] > 3:
        rgb = video[..., :3]
    else:
        rgb = video

    # Decorrelate colors
    if decorrelate:
        # Approximate color decorrelation matrix based on ImageNet statistics
        # This is a simplified version of the decorrelation used in Lucid
        color_correlation_svd_sqrt = torch.tensor([
            [0.26, 0.09, 0.02],
            [0.27, 0.00, -0.05],
            [0.27, -0.09, 0.03]
        ], device=rgb.device)

        # Reshape for matrix multiplication
        orig_shape = rgb.shape
        rgb = rgb.reshape(-1, 3)

        # Apply decorrelation
        rgb = torch.matmul(rgb, color_correlation_svd_sqrt.T)

        # Reshape back
        rgb = rgb.reshape(orig_shape)

    # Apply sigmoid to bound values
    if sigmoid:
        rgb = torch.sigmoid(rgb)

    return rgb


def video(t, w, h=None, batch=1, sd=0.01, decorrelate=True, fft=True, alpha=False, device='cuda'):
    """
    Generate a parameterized video

    Args:
        t: Number of frames
        w: Width
        h: Height (defaults to width if None)
        batch: Batch size
        sd: Standard deviation for initialization
        decorrelate: Whether to decorrelate RGB channels
        fft: Whether to use FFT parameterization
        alpha: Whether to include alpha channel
        device: Device to place tensor on

    Returns:
        Video tensor
    """
    h = h or w
    channels = 4 if alpha else 3
    shape = [batch, t, h, w, channels]

    if fft:
        param = fft_video(shape, sd=sd, device=device)
    else:
        param = torch.randn(shape, device=device) * sd

    # Process RGB channels
    rgb = to_valid_rgb(param[..., :3], decorrelate=decorrelate, sigmoid=True)

    if alpha:
        # Process alpha channel
        a = torch.sigmoid(param[..., 3:])
        return torch.cat([rgb, a], dim=-1)

    return rgb


#
# Transforms
#

class VideoTransform:
    """Base class for video transformations"""

    def __call__(self, video):
        raise NotImplementedError("Subclasses must implement this method")


class PadVideo(VideoTransform):
    """Pad a video with a constant value or reflection"""

    def __init__(self, padding, mode='reflect', value=0.5):
        self.padding = padding
        self.mode = mode
        self.value = value

    def __call__(self, video):
        # PyTorch expects [batch, channel, time, height, width]
        # But our video is [batch, time, height, width, channel]
        # So we need to permute
        video_permuted = video.permute(0, 4, 1, 2, 3)

        # Apply padding
        if self.mode == 'reflect':
            padded = F.pad(video_permuted,
                           [self.padding, self.padding, # H
                            self.padding, self.padding, # W
                            self.padding, self.padding, # T
                            0, 0, # C
                            0, 0], # B
                           mode='reflect')
        else:  # constant padding
            padded = F.pad(video_permuted,
                           [self.padding, self.padding,
                            self.padding, self.padding,
                            self.padding, self.padding,
                            0, 0,
                            0, 0],
                           mode='constant', value=self.value)

        # Permute back
        return padded.permute(0, 2, 3, 4, 1)


class JitterVideo(VideoTransform):
    """Apply random spatial jitter to video frames"""

    def __init__(self, amount):
        self.amount = amount

    def __call__(self, video):
        batch, t, h, w, c = video.shape

        # Calculate crop dimensions
        crop_h, crop_w = h - self.amount, w - self.amount

        # Random crop starting points
        b_start = torch.randint(0, self.amount + 1, (batch, t), device=video.device)
        l_start = torch.randint(0, self.amount + 1, (batch, t), device=video.device)

        # Crop each frame individually
        result = torch.zeros((batch, t, crop_h, crop_w, c), device=video.device)

        for b in range(batch):
            for f in range(t):
                top = b_start[b, f]
                left = l_start[b, f]
                result[b, f] = video[b, f, top:top + crop_h, left:left + crop_w]

        return result


class JitterVideoTemporal(VideoTransform):
    """Apply random jitter to video in all dimensions"""

    def __init__(self, amount):
        self.amount = amount

    def __call__(self, video):
        batch, t, h, w, c = video.shape

        # Calculate crop dimensions including temporal dimension
        crop_t = t - self.amount  # Crop in temporal dimension too
        crop_h = h - self.amount
        crop_w = w - self.amount

        # Random starting points for cropping
        t_start = torch.randint(0, self.amount + 1, (batch,), device=video.device)
        h_start = torch.randint(0, self.amount + 1, (batch,), device=video.device)
        w_start = torch.randint(0, self.amount + 1, (batch,), device=video.device)

        # Initialize result tensor
        result = torch.zeros((batch, crop_t, crop_h, crop_w, c), device=video.device)

        # Apply crop
        for b in range(batch):
            result[b] = video[b,
                        t_start[b]:t_start[b] + crop_t,
                        h_start[b]:h_start[b] + crop_h,
                        w_start[b]:w_start[b] + crop_w]

        return result


class RandomScaleVideo(VideoTransform):
    """Apply random scaling to video frames"""

    def __init__(self, scales):
        self.scales = scales

    def __call__(self, video):
        batch, t, h, w, c = video.shape

        # Randomly select scales
        scale_indices = torch.randint(0, len(self.scales), (batch, t), device=video.device)

        # Permute to channel-first for torch.nn.functional operations
        video_permuted = video.permute(0, 1, 4, 2, 3)

        # Initialize output tensor
        result = torch.zeros_like(video_permuted)

        for b in range(batch):
            for f in range(t):
                scale = self.scales[scale_indices[b, f].item()]
                frame = video_permuted[b, f]

                # Calculate new dimensions
                new_h, new_w = int(h * scale), int(w * scale)

                # Resize
                resized = F.interpolate(frame.unsqueeze(0), size=(new_h, new_w),
                                        mode='bilinear', align_corners=False).squeeze(0)

                # Crop or pad as needed to maintain original size
                if new_h > h:
                    # Center crop
                    start_h = (new_h - h) // 2
                    resized = resized[:, start_h:start_h + h, :]
                elif new_h < h:
                    # Pad
                    pad_h = (h - new_h) // 2
                    resized = F.pad(resized, (0, 0, pad_h, h - new_h - pad_h))

                if new_w > w:
                    # Center crop
                    start_w = (new_w - w) // 2
                    resized = resized[:, :, start_w:start_w + w]
                elif new_w < w:
                    # Pad
                    pad_w = (w - new_w) // 2
                    resized = F.pad(resized, (pad_w, w - new_w - pad_w, 0, 0))

                result[b, f] = resized

        # Permute back
        return result.permute(0, 1, 3, 4, 2)


class RandomRotateVideo(VideoTransform):
    """Apply random rotation to video frames"""

    def __init__(self, angles, units="degrees"):
        self.angles = angles
        self.units = units

    def __call__(self, video):
        batch, t, h, w, c = video.shape

        # Randomly select angles
        angle_indices = torch.randint(0, len(self.angles), (batch, t), device=video.device)

        # Permute for torch operations
        video_permuted = video.permute(0, 1, 4, 2, 3)

        # Initialize output
        result = torch.zeros_like(video_permuted)

        for b in range(batch):
            for f in range(t):
                angle = self.angles[angle_indices[b, f].item()]

                # Convert to radians if needed
                if self.units.lower() == "degrees":
                    angle = angle * np.pi / 180.0

                # Rotate the frame
                frame = video_permuted[b, f]

                # PyTorch doesn't have a direct rotation function for images/videos
                # We'd need to use a grid sampler or torchvision
                # For simplicity, we'll use torchvision's functional transforms
                frame_pil = transforms.ToPILImage()(frame.cpu())
                rotated = transforms.functional.rotate(frame_pil, angle)
                rotated_tensor = transforms.ToTensor()(rotated).to(video.device)

                result[b, f] = rotated_tensor

        # Permute back
        return result.permute(0, 1, 3, 4, 2)


class ComposeTransforms:
    """Compose multiple transforms together"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, video):
        for t in self.transforms:
            video = t(video)
        return video


# Define standard transforms
standard_transforms = ComposeTransforms([
    PadVideo(8, mode='constant', value=0.5),
    JitterVideoTemporal(8),
    RandomScaleVideo([1 + (i - 5) / 50. for i in range(11)]),
    RandomRotateVideo(list(range(-10, 11)) + 5 * [0]),
    JitterVideoTemporal(4),
])


#
# Visualization functions
#

def render_vis(model, objective_f, cfg: Config, param_f=None, optimizer_f=None,
               transforms=None, thresholds=(512,), verbose=True,
               use_fixed_seed=False, device='cuda'):
    """
    Render a visualization to maximize the given objective function

    Args:
        model: PyTorch model to visualize
        objective_f: Objective function to maximize
        param_f: Function that creates the initial parameterization
        optimizer_f: Function that creates the optimizer
        transforms: Transforms to apply to the video during optimization
        thresholds: Iterations at which to capture the current video
        verbose: Whether to show progress
        use_fixed_seed: Whether to use a fixed random seed
        device: Device to use ('cuda' or 'cpu')

    Returns:
        List of videos at the specified thresholds
    """
    if use_fixed_seed:
        torch.manual_seed(0)

    # Create initial parameterization
    if param_f is None:
        param_f = lambda: video(16, 224, batch=1, device=device)

    param = param_f().to(device)
    print(f'param.shape = {param.shape}')
    param = model_utils.to_slowfast_inputs(param.permute(0, 4, 1, 2, 3),
                                            alpha=cfg.SLOWFAST.ALPHA,
                                            requires_grad=True)
    for idx, inp in enumerate(param):
        print(f'inp_pathway_{idx}.requires_grad=', inp.requires_grad)
        print(f'inp_pathway_{idx}.shape=', inp.shape)

    # param.requires_grad_(True)

    # Create optimizer
    if optimizer_f is None:
        optimizer = optim.Adam(param, lr=0.1)
    else:
        optimizer = optimizer_f([param])

    # Ensure model is in eval mode
    # model.eval()

    # Initialize results list
    videos = []


    # Run optimization
    pbar = tqdm_notebook(range(max(thresholds) + 1))
    for i in pbar:
        optimizer.zero_grad()

        # Apply transforms to create variation
        if transforms is not None:
            transformed_param = transforms(param)
        else:
            transformed_param = param

        # Forward pass
        # inputs = model_utils.to_slowfast_inputs(
        #     transformed_param.permute(0, 4, 1, 2, 3), # To BCTHW
        #     alpha=cfg.SLOWFAST.ALPHA
        # )

        # for idx, inp in enumerate(inputs):
        #     print(f'inp_pathway_{idx}.requires_grad=', inp.requires_grad)

        _ = model(param)

        # Calculate loss (negative because we want to maximize)
        loss = -objective_f(model, transformed_param)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Save video at thresholds
        if i in thresholds:
            print(
                f"Gradient stats: mean abs: {param[0].grad.abs().mean().item()}, max: {param[0].grad.abs().max().item()}")

            print(f'mean params: {optimizer.param_groups[0]['params'][0].mean()}')
            videos.append(optimizer.param_groups[0]['params'][0].detach().cpu().permute(0, 2, 3, 4, 1))
            if verbose:
                display_videos(videos)
                print(f"Iteration {i}, Loss: {-loss.item():.4f}")

        # Update progress bar
        pbar.set_description(f"Loss: {-loss.item():.4f}")

    return videos


def render_vis_explore(model, objective_f, param_f=None, optimizer_f=None,
                       transforms=None, vis_every=100, verbose=True,
                       use_fixed_seed=False, device='cuda'):
    """
    Interactive exploration version of render_vis that can be interrupted

    Args:
        Same as render_vis, but with vis_every parameter to control
        how often to save videos

    Returns:
        List of saved videos
    """
    if use_fixed_seed:
        torch.manual_seed(0)

    # Create initial parameterization
    if param_f is None:
        param_f = lambda: video(16, 224, batch=1, device=device)

    param = param_f().to(device)
    param.requires_grad_(True)

    # Create optimizer
    if optimizer_f is None:
        optimizer = optim.Adam([param], lr=2.0)
    else:
        optimizer = optimizer_f([param])

    # Ensure model is in eval mode
    model.eval()

    # Initialize results list
    videos = []

    # Run optimization with manual progress tracking
    i = 0
    pbar = tqdm_notebook()
    try:
        while True:
            optimizer.zero_grad()

            # Apply transforms
            if transforms is not None:
                transformed_param = transforms(param)
            else:
                transformed_param = param

            # Forward pass
            model_output = model(transformed_param.permute(0, 4, 1, 2, 3))

            # Calculate loss
            loss = -objective_f(model, transformed_param)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Save video periodically
            if i % vis_every == 0:
                videos.append(param.detach().cpu())
                if verbose:
                    display_videos(videos)

            i += 1
            pbar.update(1)
            pbar.set_description(f"Iter: {i}, Loss: {-loss.item():.4f}")

    except KeyboardInterrupt:
        print(f"Optimization interrupted at iteration {i}")
        videos.append(param.detach().cpu())

    return videos


#
# Usage example
#

def example_usage():
    """Example of how to use this framework"""
    # Create model
    v = video(16, 224, batch=1, device='cpu')
    print(v.shape)
    print(v.mean())


if __name__ == '__main__':
    example_usage()
