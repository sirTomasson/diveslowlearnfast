import torch
import torch.nn as nn
from thop import profile  # For calculating FLOPs
import numpy as np

from diveslowlearnfast.config import Config
from diveslowlearnfast.models import SlowFast


def count_slowfast_gflops(model, cfg: Config):
    """
    Calculate GFLOPs for a SlowFast network.

    Args:
        model: PyTorch SlowFast model
        input_size: Input size in format (batch_size, channels, time, height, width)
        alpha: Temporal stride between slow and fast pathways (8 in the original paper)
        beta: Channel reduction ratio for fast pathway (8 in the original paper)

    Returns:
        GFLOPs value
    """
    # Create dummy input for both pathways
    batch_size = 1
    channels = 3
    time = cfg.DATA.NUM_FRAMES
    height = width = cfg.DATA.TRAIN_CROP_SIZE

    # Create input for slow pathway
    slow_input = torch.randn(batch_size, channels, time // cfg.SLOWFAST.ALPHA, height, width)

    # Create input for fast pathway
    fast_input = torch.randn(batch_size, channels, time, height, width)

    # Combine inputs for the model
    inputs = [slow_input, fast_input]

    # Use thop to profile the model
    flops, params = profile(model, inputs=(inputs,))

    # Convert to GFLOPs
    gflops = flops / 10 ** 9

    return gflops, params


# If you don't want to use thop, here's a manual approach
def count_slowfast_gflops_manual(model, cfg: Config, count_multiply_add_as_one=False):
    """
    Manually calculate GFLOPs for a SlowFast network by iterating through modules.

    Args:
        model: PyTorch SlowFast model
        count_multiply_add_as_one: If True, count a multiply-add as 1 FLOP instead of 2

    Returns:
        GFLOPs value
    """
    total_flops = 0
    module_flops = {}  # To track FLOPs per module

    # Helper function to calculate conv flops
    def calc_conv_flops(module, input_size, output_size):
        # For 3D convolution
        if isinstance(module, nn.Conv3d):
            batch_size = input_size[0]
            in_channels = input_size[1]

            out_channels = output_size[1]
            out_t, out_h, out_w = output_size[2], output_size[3], output_size[4]

            kernel_size = module.kernel_size
            if not isinstance(kernel_size, tuple):
                kernel_size = (kernel_size, kernel_size, kernel_size)
            kernel_t, kernel_h, kernel_w = kernel_size

            groups = module.groups

            # Each output element requires (kernel_t * kernel_h * kernel_w * in_channels/groups * out_channels) mul-adds
            flops = batch_size * out_t * out_h * out_w * (in_channels // groups) * out_channels * kernel_t * kernel_h * kernel_w

            # Multiply by 2 only if we're counting multiplies and adds separately
            if not count_multiply_add_as_one:
                flops *= 2

        # For 2D convolution
        elif isinstance(module, nn.Conv2d):
            batch_size = input_size[0]
            in_channels = input_size[1]

            # Handle different input dimensions for Conv2d
            if len(input_size) == 4:  # Standard 2D input
                out_channels = output_size[1]
                out_h, out_w = output_size[2], output_size[3]

                kernel_size = module.kernel_size
                if not isinstance(kernel_size, tuple):
                    kernel_size = (kernel_size, kernel_size)
                kernel_h, kernel_w = kernel_size

                groups = module.groups

                # Each output element requires (kernel_h * kernel_w * in_channels/groups * out_channels) mul-adds
                flops = batch_size * out_h * out_w * (in_channels // groups) * out_channels * kernel_h * kernel_w

                # Multiply by 2 only if we're counting multiplies and adds separately
                if not count_multiply_add_as_one:
                    flops *= 2
            else:
                # This is likely an error case, but we'll handle it gracefully
                print(f"Unexpected Conv2d input shape: {input_size}")
                flops = 0
        else:
            flops = 0

        return flops

    # Helper function to calculate linear flops
    def calc_linear_flops(module, input_size, output_size):
        # Use module properties directly
        in_features = module.in_features
        out_features = module.out_features

        # Calculate batch size by flattening all dimensions except the last one
        batch_size = 1
        for i in range(len(input_size) - 1):
            batch_size *= input_size[i]

        # Count multiply-adds
        flops = batch_size * in_features * out_features

        # Multiply by 2 only if we're counting multiplies and adds separately
        if not count_multiply_add_as_one:
            flops *= 2

        return flops

    # Register hooks to calculate flops for each module
    def add_hooks(model):
        hooks = []

        def hook_fn(module, input, output):
            nonlocal total_flops

            module_name = str(module.__class__.__name__)
            for name, mod in model.named_modules():
                if mod is module:
                    module_name = name
                    break

            if isinstance(module, (nn.Conv3d, nn.Conv2d)):
                # Make sure input is a tuple with at least one element
                if isinstance(input, tuple) and len(input) > 0:
                    flops = calc_conv_flops(module, input[0].size(), output.size())
                    total_flops += flops

                    # Store per-module flops
                    module_flops[module_name] = flops

            elif isinstance(module, nn.Linear):
                # Make sure input is a tuple with at least one element
                if isinstance(input, tuple) and len(input) > 0:
                    try:
                        flops = calc_linear_flops(module, input[0].size(), output.size())
                        total_flops += flops

                        # Store per-module flops
                        module_flops[module_name] = flops

                    except Exception as e:
                        print(f"Error in Linear layer {module_name}: {e}")
                        print(f"Input shape: {input[0].size()}")
                        print(f"Output shape: {output.size()}")
                        print(f"Module: {module}")

        # Register hook for each module
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn))

        return hooks

    # Add hooks to the model
    hooks = add_hooks(model)

    # Create dummy inputs and run a forward pass
    time = cfg.DATA.NUM_FRAMES
    height = width = cfg.DATA.TRAIN_CROP_SIZE
    # Create dummy inputs and run a forward pass
    slow_input = torch.randn(1, 3, time // cfg.SLOWFAST.ALPHA, height, width)  # Adjust based on your model's requirements
    fast_input = torch.randn(1, 3, time, height, width)  # Adjust based on your model's requirements
    inputs = [slow_input, fast_input]

    # Forward pass
    with torch.no_grad():
        try:
            model(inputs)
        except Exception as e:
            print(f"Error during forward pass: {e}")
            # Continue with what we've calculated so far

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    # Find modules with the highest FLOPs
    top_modules = sorted(module_flops.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 modules by FLOPs:")
    for module_name, flops in top_modules[:10]:
        print(f"{module_name}: {flops / 10 ** 9:.2f} GFLOPs ({flops / total_flops * 100:.2f}%)")

    # Convert to GFLOPs
    gflops = total_flops / 10 ** 9

    return gflops


# Example usage
def main():
    # Load your SlowFast model
    # model = SlowFast(cfg)  # Your code to load the model

    num_frames = [16, 32, 64, 128]

    cfg = Config()
    cfg.DATA_LOADER.BATCH_SIZE = 1
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.SLOWFAST.ALPHA = 8
    cfg.MODEL.NUM_CLASSES = 48
    # Option 1: Using thop
    for time in  num_frames:
        cfg.DATA.NUM_FRAMES = time
        model = SlowFast(cfg)
        print(f'Calculating GFLOPs $T$={time}')

        gflops, params = count_slowfast_gflops(model, cfg=cfg)
        print(f"Model GFLOPs: {gflops:.2f}")
        print(f"Model Parameters: {params / 10 ** 6:.2f}M")

        # Option 2: Using manual counting with multiply-add as 2 FLOPs
        try:
            gflops_manual = count_slowfast_gflops_manual(model, cfg, count_multiply_add_as_one=False)
            print(f"Model GFLOPs (manual, counting multiply-add as 2 FLOPs): {gflops_manual:.2f}")
        except Exception as e:
            print(f"Error using manual counting: {e}")

        # Option 3: Using manual counting with multiply-add as 1 FLOP
        try:
            gflops_manual_one = count_slowfast_gflops_manual(model, cfg, count_multiply_add_as_one=True)
            print(f"Model GFLOPs (manual, counting multiply-add as 1 FLOP): {gflops_manual_one:.2f}")
        except Exception as e:
            print(f"Error using manual counting: {e}")


if __name__ == "__main__":
    main()