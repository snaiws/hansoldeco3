from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline

import torch
from langchain_community.llms.vllm import VLLM


class LLM_loading_cases:
    '''
    유인원식 클래스
    '''
    def load_base(model_name: str = "NCSOFT/Llama-VARCO-8B-Instruct", max_new_tokens = 64, temperature = 0.1, top_p = 1, top_k = -1):
        """
        허깅페이스 모델 그대로 쓰는 함수
        추후 훈련 고려해서 변경
        """
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=temperature,
            top_p = top_p,
            top_k = top_k,
            return_full_text=False,
            max_new_tokens=max_new_tokens
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        return llm



    def load_vllm(model_name: str = "NCSOFT/Llama-VARCO-8B-Instruct", max_new_tokens = 64, temperature = 0.1, top_p = 1, top_k = -1):
        llm = VLLM(
            model=model_name,
            temperature=temperature,
            top_p = top_p,
            top_k = top_k,
            max_new_tokens=max_new_tokens,
            dtype="float16",  # or torch.float16, for FP16 precision&#8203;:contentReference[oaicite:2]{index=2}
            vllm_kwargs={     # extra vLLM.LLM options passed through to the vLLM engine
                "quantization": "bitsandbytes",      # use 4-bit bitsandbytes quantization&#8203;:contentReference[oaicite:3]{index=3}
                "load_format": "bitsandbytes",       # load weights in BitsAndBytes 4-bit format&#8203;:contentReference[oaicite:4]{index=4}
                "gpu_memory_utilization": 0.9,
                # "trust_remote_code": True  # enable if the model requires remote code execution
            }
        )

        return llm
    

def load_LLM(case = "load_vllm", control_params = {""}):
    '''
    control_params는 일단 llm 출력에 영향주는
    model_name
    temperature
    top_p 
    top_k
    max_new_tokens
    '''
    return getattr(LLM_loading_cases, case)(**control_params)