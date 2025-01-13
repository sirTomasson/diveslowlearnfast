from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConfigData:
    INPUT_CHANNEL_NUM: list[int] = field(
        default_factory=lambda: [3, 3]
    )
    NUM_FRAMES: int = 32
    SAMPLING_RATE: int = 2
    TRAIN_JITTER_SCALES: list[int] = field(
        default_factory=lambda: [256, 320]
    )
    TRAIN_CROP_SIZE: int = 224
    TEST_CROP_SIZE: int = 256
    DATASET_PATH: Path = None,
    MEAN: tuple[float, float, float] = field(default_factory=lambda: (0.31, 0.47, 0.5))
    STD: tuple[float, float, float] = field(default_factory=lambda: (0.2, 0.2, 0.23))
    TEMPORAL_RANDOM_JITTER: int = 0
    TEMPORAL_RANDOM_OFFSET: int = 0
    MULTI_THREAD_DECODE: bool = False
    THRESHOLD: int = -1

@dataclass
class RandAugmentConfig:
    ENABLED: bool = False
    PROB: float = 0.5
    NUM_LAYERS: int = 2
    MAGNITUDE: int = 9

@dataclass
class RandomRotateConfig:
    ENABLED: bool = False
    MAX_DEGREE: int = 30


@dataclass
class SlowFastConfig:
    ALPHA: int = 8
    BETA_INV: int = 8
    FUSION_CONV_CHANNEL_RATIO: int = 2
    FUSION_KERNEL_SZ: int = 7


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
    GLOBAL_SYNC: bool = False
    NUM_SYNC_DEVICES: int = 1


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
    LONG_CYCLE: bool = False
    SHORT_CYCLE_FACTORS: list[float] = field(default_factory=lambda: [0.5, 0.5 ** 0.5])
    LONG_CYCLE_FACTORS: list[tuple[float]] = field(default_factory=lambda: [
        (0.25, 0.5 ** 0.5),
        (0.5, 0.5 ** 0.5),
        (0.5, 1),
        (1, 1),
    ])
    EPOCH_FACTOR: float = 1.5
    BN_BASE_SIZE: int = 8
    SHORT_CYCLE_PERIOD: int = 3


@dataclass
class TrainConfig:
    ENABLED: bool = True
    BATCH_SIZE: int = 4
    MACRO_BATCH_SIZE: int = 256
    CHECKPOINT_PERIOD: int = 10
    CHECKPOINT_FILENAME: str = ''
    EVAL_PERIOD: int = 10
    AUTO_RESUME: bool = True
    RESULT_DIR: Path = Path('results')
    WEIGHTS_PATH: str = ''


@dataclass
class EvalConfig:
    ENABLED: bool = False
    RESULT_DIR: Path = Path('results/eval')


@dataclass
class SolverConfig:
    BASE_LR: int = 0.1
    MAX_EPOCH: int = 196
    MOMENTUM: int = 0.9
    WEIGHT_DECAY: int = 1e-4
    WARMUP_EPOCHS: int = 34
    WARMUP_START_LR: int = 0.01
    OPTIMIZING_METHOD: str = 'sgd'
    STEPS: list[int] = field(default_factory=lambda: [0, 94, 154, 196])
    GAMMA: float = 0.1


@dataclass
class DataLoaderConfig:
    NUM_WORKERS: int = 8
    PIN_MEMORY: bool = True
    USE_DECORD: bool = False


@dataclass
class Config:
    NUM_GPUS: int = 1
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
    RAND_AUGMENT: RandAugmentConfig = field(default_factory=RandAugmentConfig)
    RANDOM_ROTATE: RandomRotateConfig = field(default_factory=RandomRotateConfig)
    EVAL: EvalConfig = field(default_factory=EvalConfig)
