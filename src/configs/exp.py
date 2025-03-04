from dataclasses import dataclass, asdict



@dataclass
class Experiments:
    model_name : str = "NCSOFT/Llama-VARCO-8B-Instruct"
    chain_type = ["stuff", "stuff"]
    data_prompt_template : str = "exp_0"
    data_pipeline : str =  "exp_0"
    data_encoding : str = "utf-8-sig"
    data_chunk_size : int = 1024
    RAG_chain_type1 = "stuff"
    RAG_chain_type2 = "stuff"

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
    
    
    print(config.model_name)
    print(config.chain_type)
    print(config.data_prompt_template)
    print(config.data_pipeline)
    print(config.data_encoding)
    print(config.data_chunk_size)
    print(config.RAG_chain_type1)
    print(config.RAG_chain_type2)