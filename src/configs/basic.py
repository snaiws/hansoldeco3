import os
from dataclasses import dataclass, asdict, field

import torch


@dataclass
class EnvConfig:
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = field(default_factory=lambda: os.cpu_count() // 2)
    use_amp: bool = field(default_factory=lambda: torch.cuda.is_available())  # Automatic Mixed Precision
    mixed_precision: str = "fp16"  # "bf16"도 가능

    def __post_init__(self):
        print(f"Using device: {self.device}, num_workers: {self.num_workers}, AMP: {self.use_amp}")
