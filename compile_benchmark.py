import logging
import os

import torch
import numpy as np

from diveslowlearnfast.config import Config
from diveslowlearnfast.models import SlowFast
from diveslowlearnfast.train import helper as train_helper

logger = logging.getLogger(__name__)


def set_log_level():
    logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def init_model(cfg, device):
    return SlowFast(cfg).to(device)


def generate_data(n_samples):
    # B x C x T x H x W
    xb = torch.randn((n_samples, 3, 16, 224, 224))
    yb = torch.randn((n_samples, 48))
    return xb, yb

def main():
    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True

    if not gpu_ok:
        logger.warning(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the name of the current GPU device
        print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    else:
        print("No GPU available")

    cfg = Config()
    cfg.SLOWFAST.ALPHA = 4
    cfg.DATA.DATASET_PATH = '/home/s2871513/Datasets/Diving48'
    N_ITERS = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_model(cfg, device)
    criterion, optimiser, *_ = train_helper.get_train_objects(cfg, model)

    def train(mod, xb, yb):
        optimiser.zero_grad(True)
        pred = train_helper.forward(mod, xb, device, cfg)
        loss = criterion(pred, yb)
        loss.backward()
        optimiser.step()

    eager_times = []
    for i in range(N_ITERS):
        xb, yb = generate_data(16)
        train_helper.forward(model, xb, device, cfg)
        print(xb.shape, yb.shape)
        _, eager_time = timed(lambda: train(model, xb, yb))
        eager_times.append(eager_time)
        print(f"eager train time {i}: {eager_time}")
    print("~" * 10)

    model = init_model(cfg, device)
    train_opt = torch.compile(train, mode="reduce-overhead")

    compile_times = []
    for i in range(N_ITERS):
        xb, yb = generate_data(16)
        _, compile_time = timed(lambda: train(train_opt, xb, yb))
        compile_times.append(compile_time)
        print(f"compile train time {i}: {compile_time}")
    print("~" * 10)

    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    assert (speedup > 1)
    print(f"(train) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
    print("~" * 10)

if __name__ == '__main__':
    main()