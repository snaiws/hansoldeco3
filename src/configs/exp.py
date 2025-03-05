from dataclasses import dataclass, asdict



@dataclass
class ExpDefineUnit:
    '''
    실험 파라미터
    data / prompt / RAG / model로 나누어 관리
    '''
    train: str = 'raw/train.csv'
    test: str =  'raw/test.csv'
    data_encoding : str = "utf-8-sig"
    data_pipeline : str =  "pipeline_0"
    prompt_template : str = "exp_0"
    RAG_chain_type1 = "stuff"
    RAG_chain_type2 = "stuff"
    model_name : str = "NCSOFT/Llama-VARCO-8B-Instruct"

    def to_dict(self):
        asdict(self)

    @classmethod
    def exp_0(cls):
        return cls()
    
    @classmethod
    def exp_1(cls):
        return cls(train = "sample/v1/train.csv", test = "sample/v1/test.csv")


def build_exp(exp_name = "exp_0"):
    return getattr(ExpDefineUnit, exp_name)()



if __name__ == "__main__":
    config = build_exp()
    
    
    print(config.model_name)
    print(config.prompt_template)
    print(config.data_pipeline)
    print(config.data_encoding)
    print(config.data_chunk_size)
    print(config.RAG_chain_type1)
    print(config.RAG_chain_type2)