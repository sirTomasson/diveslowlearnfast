from itertools import cycle, islice

import numpy as np

from tqdm import tqdm
from diveslowlearnfast.config import Config
from diveslowlearnfast.train import helper as train_helper

from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):

    def __init__(self, optimizer, start_lr, end_lr, last_epoch=-1):
        self.schedule =  np.linspace(start_lr, end_lr, last_epoch)
        self.initial_lr = start_lr
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule[self._step_count] for _ in self.base_lrs]

def run_warmup(model, optimiser, criterion, dataloader, device, cfg: Config):
    warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
    lr_schedule = np.linspace(cfg.SOLVER.WARMUP_START_LR, cfg.SOLVER.BASE_LR, warmup_epochs)
    warmup_bar = tqdm(range(warmup_epochs))
    dataloader_iter = islice(cycle(dataloader), warmup_epochs)
    warmup_bar.set_description("Warming up")
    for i in warmup_bar:
        xb, yb, _, masks_slow, masks_fast = train_helper.get_batch(dataloader_iter, device, data_requires_grad=cfg.EGL.ENABLED)

        lr = lr_schedule[i]
        optimiser.param_groups[0]['lr'] = lr
        warmup_bar.set_postfix({ 'lr': lr })
        optimiser.zero_grad()

        logits = train_helper.forward(model, xb, device, cfg)
        if cfg.EGL.ENABLED:
            loss, _ = criterion(logits, yb, xb, [masks_slow, masks_fast], warmup=True)
        else:
            loss = criterion(logits, yb)

        loss.backward()
        optimiser.step()
