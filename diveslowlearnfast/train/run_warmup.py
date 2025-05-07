
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from diveslowlearnfast.config import Config
from diveslowlearnfast.train import helper as train_helper
from itertools import cycle, islice

def run_warmup(model, optimiser, dataloader, device, cfg: Config):
    warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
    lr_schedule = np.linspace(cfg.SOLVER.WARMUP_START_LR, cfg.SOLVER.BASE_LR, warmup_epochs)
    warmup_bar = tqdm(range(warmup_epochs))
    dataloader_iter = islice(cycle(dataloader), warmup_epochs)
    warmup_bar.set_description("Warming up")
    for i in warmup_bar:
        xb, yb, *_ = train_helper.get_batch(dataloader_iter, device)

        lr = lr_schedule[i]
        optimiser.param_groups[0]['lr'] = lr
        warmup_bar.set_postfix({ 'lr': lr })
        optimiser.zero_grad()

        logits = train_helper.forward(model, xb, device, cfg)
        loss = F.cross_entropy(logits, yb)

        loss.backward()
        optimiser.step()
