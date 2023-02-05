from dataclasses import dataclass, field


@dataclass
class CFG:
    SEED: int = 42
    IMG_HEIGHT: int = 1344
    IMG_WIDTH: int = 768
    N_CHANNELS: int = 1
    N_SAMPLES_RECORD: int = 548

    LR_MAX: float = 5e-6
    WD_RATIO: float = 0.01
    WARMUP_EPOCHS: int = 0
    EPOCHS: int = 10
    BATCH_SIZE: int = 8
    