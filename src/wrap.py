import os

import pandas as pd

from wrapup.evaluation import calculate_similarities, scoring
from configs import EnvDefineUnit, build_exp


exp = "exp_3"
env = EnvDefineUnit()
config_exp = build_exp(exp)
encoding = config_exp.data_encoding


# 경로 - path manager 추상화 실패
path_train = os.path.join(env.PATH_DATA_DIR, config_exp.train)
pred = pd.read_csv("/workspace/Storage/hansoldeco3/Log/exp/2025-03-11 20:56:54/result.csv")
true = pd.read_csv(path_train, encoding = encoding)

cossims, jaccardsims = calculate_similarities(true['재발방지대책 및 향후조치계획'], pred['answer'])
score = scoring(cossims, jaccardsims)
print(score)

df = pd.concat([true[['ID', '재발방지대책 및 향후조치계획']], pred['answer']], axis=1)
df.columns = ['ID','true', 'pred']
df.to_csv("result.csv", index=False)
print(df)
