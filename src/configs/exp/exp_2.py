
from .exp_base import ExpDefineUnit  # 별도 파일에 선언된 dataclass

def get_exp():
    return ExpDefineUnit(
        exp_name = "exp_2",
        train = "sample/v1/train.csv", 
        test = "sample/v1/test.csv",
        chain_strategy = "dual", # 주요변경
        version_prompt_precendent = "exp_1", # 주요변경
        prompt_template_format = "exp_1", # 주요변경
        retriever_guideline_name = "FAISSVSUnit",
        embedding_model_name_guideline = "jhgan/ko-sbert-nli",
        retriever_guideline_params = (
            {
                "search_type" : "similarity",
                "search_kwargs" : {
                    "k" : 5
                }
            },
        ),
        splitter_guideline_name = "RecursiveCharacterTextSplitter",
        splitter_guideline_kwargs = {"chunk_size":100, "chunk_overlap" : 20},
        max_new_tokens = 200 # 주요변경
    )