
from .exp_base import ExpDefineUnit  # 별도 파일에 선언된 dataclass

def get_exp():
    return ExpDefineUnit(
        train = "sample/v1/train.csv", 
        test = "sample/v1/test.csv"
        ) # 주요변경