import os
from dataclasses import dataclass, asdict, field

import torch

@dataclass
class PathConfig:
    data_train_raw: str = "/raw/train.csv"
    data_test_raw: str = "/raw/test.csv"
    data_submission: str = "/raw/sample_submission.csv"
    dir_guidelines: str = "/raw/건설안전지침"

@dataclass
class EnvConfig:
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = field(default_factory=lambda: os.cpu_count() // 2)
    use_amp: bool = field(default_factory=lambda: torch.cuda.is_available())  # Automatic Mixed Precision
    mixed_precision: str = "fp16"  # "bf16"도 가능

    def __post_init__(self):
        print(f"Using device: {self.device}, num_workers: {self.num_workers}, AMP: {self.use_amp}")

@dataclass
class Experiments:
    model_name : str = "NCSOFT/Llama-VARCO-8B-Instruct"
    chain_type = ["stuff", "stuff"]
    prompt_template : str = "prompt_template_0"
    pipeline : str =  "pipeline_0"
    
    env: EnvConfig = field(default_factory=EnvConfig)  # 환경 설정
    path: PathConfig = field(default_factory=PathConfig)  # 경로 설정

    def to_dict(self):
        asdict(self)

    @classmethod
    def exp_0(cls):
        return cls()
    
    @classmethod
    def exp_1(cls):
        return cls(model_name = "")

def build_exp(exp_name = "exp_0"):
    return getattr(Experiments, exp_name)()


if __name__ == "__main__":
    config = build_exp()

    # 경로 출력
    print(config.path.data_train_raw)  # /raw/train.csv
    print(config.path.dir_guidelines)  # /raw/건설안전지침



    
    # configs
    csv_path = 
    pdf_files = 
    chunk_size = 1024
    encoding = 'utf-8-sig'

    chain_type1 = "stuff"
    chain_type2 = "stuff"
    model_name = "NCSOFT/Llama-VARCO-8B-Instruct"
    id_prompt_template = "exp_0"
    pipeline = "exp_0"