from langchain_core.runnables import Runnable
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable, RunnableLambda


from vllm import LLM, SamplingParams


# vLLM을 LangChain 호환 `Runnable`로 감싸는 클래스
class VLLMRunner(Runnable, BaseModel):
    llm: LLM

    class Config:
        arbitrary_types_allowed = True  # ✅ vllm.LLM 객체 허용

    def invoke(self, input_text: Any, config=None, **kwargs) -> str:
        """vLLM을 호출하여 응답을 반환하는 함수"""
        # ✅ 입력이 딕셔너리라면 문자열로 변환
        if isinstance(input_text, dict):
            input_text = input_text.get("question", "")

        input_text = str(input_text)  # 문자열로 강제 변환

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 512),
            stop=kwargs.get("stop", None),
        )
        outputs = self.llm.generate([input_text], sampling_params)
        return outputs[0].outputs[0].text  # 생성된 텍스트 반환