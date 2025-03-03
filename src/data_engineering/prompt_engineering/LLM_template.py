from dataclasses import dataclass, asdict


'''
context와 question을 가지고 최종적으로 LLM에 입력할 프롬프트
'''


@dataclass
class Template:
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

    def to_dict(self):
        asdict(self)

    @classmethod
    def exp_0(cls):
        return cls


def get_prompt_template(exp = "exp_0"):
    return getattr(Template, exp)
