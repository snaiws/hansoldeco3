from dataclasses import dataclass, asdict, field



@dataclass
class ExpDefineUnit:
    '''
    실험 파라미터
    chain / data / prompt / RAG / model로 나누어 관리
    '''
    # chain 파라미터
    chain_strategy : str = "baseline"
    chain_type :str = "stuff"

    # data 파라미터
    train: str = 'raw/train.csv'
    test: str =  'raw/test.csv'
    data_encoding : str = "utf-8-sig"
    data_pipeline : str =  "pipeline_0"

    # prompt 파라미터
    prompt_template_format : str = "exp_0"
    version_prompt_precendent : str = "exp_0"
    version_prompt_question : str = "exp_0"

    # RAG 파라미터
    embedding_model_name_precendent :str = "jhgan/ko-sbert-nli"

    retriever_precendent_name : str = "FAISSVSUnit"
    retriever_precendent_params : tuple = field(default_factory=lambda: (
        (
            {
                "search_type" : "similarity",
                "search_kwargs" : {
                    "k" : 5
                }
            },
        )
        ))
    splitter_precendent_name : str = None
    splitter_precendent_kwargs : dict = None

    embedding_model_name_guideline :str = None
    retriever_guideline_name : str = None
    retriever_guideline_params : tuple = None
    splitter_guideline_name : str = None
    splitter_guideline_kwargs : dict = None

    # model 파라미터
    model_strategy : str = 'load_vllm'
    model_name : str = "NCSOFT/Llama-VARCO-8B-Instruct"
    temperature : float = 0.1
    top_p : float = 1.0
    top_k : float = -1
    max_new_tokens : int = 64


    def to_dict(self):
        return asdict(self)

    @classmethod
    def exp_0(cls):
        return cls()
    
    @classmethod
    def exp_1(cls):
        return cls(train = "sample/v1/train.csv", test = "sample/v1/test.csv")

    @classmethod
    def exp_2(cls):
        return cls(
            train = "sample/v1/train.csv", 
            test = "sample/v1/test.csv",
            chain_strategy = "",
            prompt_template_format = "",
            retriever_guideline_name = "FAISSVSUnit",
            embedding_model_name_guideline = "jhgan/ko-sbert-nli",
            retriever_guideline_params = field(default_factory=lambda: 
                (
                    {
                        "search_type" : "similarity",
                        "search_kwargs" : {
                            "k" : 5
                        }
                    }
                )),
            splitter_guideline_name = "RecursiveCharacterTextSplitter",
            splitter_guideline_kwargs = field(default_factory=lambda: {"chunk_size":100, "chunk_overlap" : 20})
        )

def build_exp(exp_name = "exp_0"):
    return getattr(ExpDefineUnit, exp_name)()



if __name__ == "__main__":
    config = build_exp()
    
    
    print(config.model_name)