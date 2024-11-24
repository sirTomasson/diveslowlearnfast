from itertools import cycle, islice

import numpy as np

from tqdm import tqdm
from diveslowlearnfast.config import Config



def run_warmup(model, optimiser, criterion, dataloader, device, cfg: Config):
    warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
    lr_schedule = np.linspace(cfg.SOLVER.WARMUP_START_LR, cfg.SOLVER.BASE_LR, warmup_epochs)
    warmup_bar = tqdm(enumerate(islice(cycle(dataloader), warmup_epochs)))
    warmup_bar.set_description("Warming up")
    for i, batch in warmup_bar:
        xb, yb = batch[0], batch[1]
        lr = lr_schedule[i]
        warmup_bar.set_postfix({ 'lr': lr })

        yb = yb.to(device)
        xb_fast = xb[:].to(device)
        # reduce the number of frames by the alpha ratio
        # B x C x T / alpha x H x W
        xb_slow = xb[:, :, ::cfg.SLOWFAST.ALPHA].to(device)

        optimiser.zero_grad()
        o = model([xb_slow, xb_fast])
        loss = criterion(o, yb)
        loss.backward()

