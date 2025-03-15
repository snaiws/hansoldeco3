from dataclasses import dataclass

from .exp_1 import Experiment_1  # 별도 파일에 선언된 dataclass



@dataclass    
class Experiment_2(Experiment_1):
    exp_name : str = "exp_2",
    chain_strategy : str = "dual", # 주요변경
    version_prompt_precendent : str = "exp_1", # 주요변경
    prompt_template_format : str = "exp_1", # 주요변경
    retriever_guideline_name : str = "FAISSVSUnit",
    embedding_model_name_guideline : str = "jhgan/ko-sbert-nli",
    retriever_guideline_params : tuple = (
        {
            "search_type" : "similarity",
            "search_kwargs" : {
                "k" : 5
            }
        },
    ),
    splitter_guideline_name : str = "RecursiveCharacterTextSplitter",
    splitter_guideline_kwargs : dict = {"chunk_size":100, "chunk_overlap" : 20},
    max_new_tokens : int = 200 # 주요변경