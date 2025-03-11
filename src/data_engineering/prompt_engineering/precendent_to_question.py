import pandas as pd



class Question:
    def exp_0(row):
        question = (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"대책은 무엇인가요?"
            )
        return question
    
    def exp_1(row):
        question = (
            # 시간
            f"다음과 같은 상황에서 공사 중 사고가 발생했습니다."
            f"계절: '{'모름' if pd.isna(row['계절']) else row['계절']}',"
            f"시간대: '{'모름' if pd.isna(row['시간대']) else row['시간대']}',"
            f"날씨: '{'모름' if pd.isna(row['날씨']) else row['날씨']}',"
            f"기온: '{'모름' if pd.isna(row['기온']) else row['기온']}',"
            f"습도: '{'모름' if pd.isna(row['습도']) else row['습도']}',"
            f"장소: '{'모름' if pd.isna(row['장소']) else row['장소']}'"
            f"지상 '{'모름' if pd.isna(row['지상']) else str(row['지상'])+'층'}', 지하 '{'모름' if pd.isna(row['지하']) else str(row['지하'])+'층'}'인 구조물 공사에서 사고 발생."

            f"공사종류: 대분류 '{'모름' if pd.isna(row['공사종류_대분류']) else row['공사종류_대분류']}', 중분류 '{'모름' if pd.isna(row['공사종류_중분류']) else row['공사종류_중분류']}', 소분류 '{'모름' if pd.isna(row['공사종류_소분류']) else row['공사종류_소분류']}' 공사 중 "
            f"공종 대분류: '{row['공종_대분류']}', 중분류 '{row['공종_소분류']}' 작업에서 "
            f"작업 프로세스느 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"사고가 발견되기까지 걸린 시간은 '{'모름' if pd.isna(row['사고인지시차']) else row['사고인지시차']}',"
            
            f"인적피해는 '{'모름' if pd.isna(row['인적사고']) else row['인적사고']}',"
            f"피해 당시 근무종류: '{'모름' if pd.isna(row['근무종류']) else row['근무종류']}',"
            f"물적피해는 '{'' if pd.isna(row['사고객체_대분류']) else row['사고객체_대분류']+'인 '}{'' if pd.isna(row['사고객체_소분류']) else row['사고객체_소분류']+'가 '}{'' if pd.isna(row['물적사고']) else row['물적사고']}',"
            f"사고부위: '{'' if pd.isna(row['부위2']) else pd.isna(row['부위2'])}'"
            
            f"사고객체 '{row['사고객체_대분류']}'(소분류: '{row['사고객체_소분류']}')와 관련된 사고가 발생했습니다. "
            
            f"이에 대한 '대책'은 무엇인가요?"
            )
        return question




def get_prompt_question(data, exp:str = "exp_0", kwargs:dict = {}):
    return getattr(Question, exp)(data, **kwargs)