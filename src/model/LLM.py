from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline

##############################################
# 4. LLM 모델 로드
##############################################
def load_llm_model(model_name: str = "NCSOFT/Llama-VARCO-8B-Instruct", bnb_config = None):
    """
    허깅페이스 모델 그대로 쓰는 함수
    추후 훈련 고려해서 변경
    """

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
