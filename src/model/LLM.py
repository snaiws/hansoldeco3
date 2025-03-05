from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline

import torch
from vllm import LLM



def load_llm_model_huggingface(model_name: str = "NCSOFT/Llama-VARCO-8B-Instruct"):
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
        temperature=0.1,
        return_full_text=False,
        max_new_tokens=64
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm



def load_llm_model_vllm(model_name: str = "NCSOFT/Llama-VARCO-8B-Instruct"):
    # vLLM에서 bitsandbytes 8bit 적용
    llm = LLM(
        model=model_name,
        quantization="bitsandbytes",
        load_format="bitsandbytes",  # 이 옵션 추가
        dtype=torch.float16,
        gpu_memory_utilization=0.9  # GPU 메모리 활용율 조정 가능
    )
    return llm