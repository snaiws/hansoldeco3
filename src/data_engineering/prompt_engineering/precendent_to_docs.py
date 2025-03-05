import pandas as pd

class DF_to_docs:
    def exp_0(row):
        q1 = (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
            )
        a1 = row["재발방지대책 및 향후조치계획"]
        precendent = f"Q: {q1}\nA: {a1}"
        return precendent


def get_prompt_precendent(data, exp:str = "exp_0", kwargs:dict = {}):
    return getattr(DF_to_docs, exp)(data, **kwargs)
