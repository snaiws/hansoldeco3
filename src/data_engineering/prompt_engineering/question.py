from dataclasses import dataclass

from langchain.prompts import PromptTemplate

@dataclass
class SafetyPromptConfig:
    """
    - 프롬프트 템플릿을 포함해, 프롬프트 구성에 필요한 필드들을 dataclass로 선언.
    - 실험 시 template 등을 다양하게 바꿔주면 됨.
    """
    template: str = """
### 지침: 당신은 건설 안전 전문가입니다.
질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 표현을 포함하지 마세요.

{context}

### 질문:
{question}

[/INST]
"""
    max_tokens: int = 64
    temperature: float = 0.1
    # 추가로 필요한 인자(예: top_k, top_p 등)를 자유롭게 정의 가능
    # ...



def build_prompt_template():
    """
    RAG 체인에 들어갈 프롬프트 템플릿을 생성한다.
    필요하면 추가/수정 가능
    """
    print("2. 프롬프트 템플릿을 생성합니다.")
    prompt_template = """
### 지침: 당신은 건설 안전 전문가입니다.
질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 표현을 포함하지 마세요.

{context}

### 질문:
{question}

[/INST]
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    print(" - 프롬프트 템플릿 생성 완료!")
    return prompt