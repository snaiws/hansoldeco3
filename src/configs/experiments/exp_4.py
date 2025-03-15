from dataclasses import dataclass

from .exp_3 import Experiment_3  # 별도 파일에 선언된 dataclass



@dataclass    
class Experiment_4(Experiment_3):
    exp_name : str = "exp_4",
    data_pipeline : str = "pipeline_2", # 주요변경
    version_prompt_precendent : str = "exp_2", # 주요변경
    version_prompt_question : str = "exp_1", # 주요변경
    repetition_penalty : float = 1.0,
    frequency_penalty : float = 2.0,
    presence_penalty : float = 0.1,