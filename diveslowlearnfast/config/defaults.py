from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConfigData:
    INPUT_CHANNEL_NUM: list[int] = field(
        default_factory=lambda: [3, 3]
    )
    NUM_FRAMES = 32
    SAMPLING_RATE = 2
    TRAIN_JITTER_SCALES: list[int] = field(
        default_factory=lambda: [256, 320]
    )
    TRAIN_CROP_SIZE = 224
    TEST_CROP_SIZE = 256
    ANNOTATIONS_PATH: Path = None,
    VIDEOS_PATH: Path = None,

@dataclass
class SlowFastConfig:
    ALPHA = 8
    BETA_INV = 8
    FUSION_CONV_CHANNEL_RATIO = 2
    FUSION_KERNEL_SZ = 7

@dataclass
class ResNetConfig:
    ZERO_INIT_FINAL_BN: bool = True
    WIDTH_PER_GROUP: int = 64
    NUM_GROUPS: int = 1
    DEPTH: int = 50
    TRANS_FUNC: str = 'bottleneck_transform'
    STRIDE_1X1: bool = False
    NUM_BLOCK_TEMP_KERNEL: list[list[int]] = field(
        default_factory=lambda: [[3, 3], [4, 4], [6, 6], [3, 3]]
    )
    SPATIAL_STRIDES: list[list[int]] = field(
        default_factory=lambda: [[1, 1], [2, 2], [2, 2], [2, 2]]
    )
    SPATIAL_DILATIONS: list[list[int]] = field(
        default_factory=lambda: [[1, 1], [1, 1], [1, 1], [1, 1]]
    )
    ZERO_INIT_FINAL_CONV: bool = False

@dataclass
class NonLocalConfig:
    LOCATION: list[list[list]] = field(
        default_factory=lambda: [[[], []], [[], []], [[], []], [[], []]]
    )
    GROUP: list[list[int]] = field(
        default_factory=lambda: [[1, 1], [1, 1], [1, 1], [1, 1]]
    )
    INSTANTIATION: str = 'dot_product'
    POOL: list[list[list[int]]] = field(
        default_factory=lambda: [
            [[1, 2, 2], [1, 2, 2]],
            [[1, 2, 2], [1, 2, 2]],
            [[1, 2, 2], [1, 2, 2]],
            [[1, 2, 2], [1, 2, 2]],
        ]
    )

@dataclass
class BNConfig:
    NORM_TYPE: str = 'batchnorm'
    USE_PRECISE_STATS: bool = True
    NUM_BATCHES_PRECISE: int = 200

@dataclass
class ModelConfig:
    NUM_CLASSES: int = 48
    ARCH: str = 'slowfast'
    MODEL_NAME: str = 'SlowFast'
    LOSS_FUNC: str = 'cross_entropy'
    DROPOUT_RATE: float = 0.5
    HEAD_ACT: str = 'softmax'
    DETACH_FINAL_FC: bool = False
    FC_INIT_STD: float = 0.01

@dataclass
class DetectionConfig:
    ENABLE: bool = False

@dataclass
class MultiGridConfig:
    SHORT_CYCLE: bool = False

@dataclass
class TrainConfig:
    BATCH_SIZE: int = 4

class SolverConfig:
    BASE_LR = 0.1
    MAX_EPOCH = 196
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 34.0
    WARMUP_START_LR = 0.01
    OPTIMIZING_METHOD = 'sgd'


@dataclass
class DataLoaderConfig:
    NUM_WORKERS = 8
    PIN_MEMORY = True

@dataclass
class Config:
    DATA: ConfigData = field(default_factory=ConfigData)
    SLOWFAST: SlowFastConfig = field(default_factory=SlowFastConfig)
    RESNET: ResNetConfig = field(default_factory=ResNetConfig)
    NONLOCAL: NonLocalConfig = field(default_factory=NonLocalConfig)
    BN: BNConfig = field(default_factory=BNConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    DETECTION: DetectionConfig = field(default_factory=DetectionConfig)
    MULTIGRID: MultiGridConfig = field(default_factory=MultiGridConfig)
    TRAIN: TrainConfig = field(default_factory=TrainConfig)
    SOLVER: SolverConfig = field(default_factory=SolverConfig)
    DATA_LOADER: DataLoaderConfig = field(default_factory=DataLoaderConfig)
