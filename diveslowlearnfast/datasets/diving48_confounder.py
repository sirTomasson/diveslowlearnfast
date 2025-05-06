from torch.utils.data import Dataset

from diveslowlearnfast.config import Config
from diveslowlearnfast.datasets.diving48 import Diving48Dataset
from diveslowlearnfast.datasets.superimpose_confounder import superimpose_confounder


class Diving48ConfounderDatasetWrapper(Dataset):
    def __init__(
            self,
            diving48: Diving48Dataset,
            cfg: Config,
    ):
        self.diving48 = diving48
        self.size = cfg.CONFOUNDERS.SIZE
        self.grid_size = cfg.CONFOUNDERS.GRID_SIZE
        self.channel = cfg.CONFOUNDERS.CHANNEL
        self.inplace = cfg.CONFOUNDERS.INPLACE

    def __getitem__(self, idx):
        frames, label, io_time, transform_time, video_id, masks = self.diving48[idx]
        frames = superimpose_confounder(
            frames,
            label,
            self.size,
            self.grid_size,
            self.channel,
            self.inplace
        )
        return frames, label, io_time, transform_time, video_id, masks

    def __len__(self):
        return len(self.diving48)
