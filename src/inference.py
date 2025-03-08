import os
import json

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
from tqdm import tqdm

from configs import build_exp, EnvDefineUnit
from data_engineering.dataset.precendent import CSVPrecendentDataset, install_pipeline
from data_engineering.prompt_engineering.LLM_template import get_prompt_template
from data_engineering.prompt_engineering.precendent_to_docs import get_prompt_precendent
from data_engineering.prompt_engineering.precendent_to_question import get_prompt_question
from data_engineering.dataset.guideline import PDFDataset
from data_engineering.RAG import build_vectorstore
from model.LLM import load_LLM
from utils.now import get_now



def infer_baseline_1():
    """
    (1) 선례 데이터 전처리 후 선례 docs 생성, 선례 데이터 
    (2) DF-기반 Retrievr와 PDF-기반 Retriever 중 선택(또는 병합) 가능
    (3) 결과를 CSV로 저장
    """
    env = EnvDefineUnit()
    config_exp = build_exp('exp_1')

    now = get_now("Asia/Seoul")


    # 경로
    path_train = os.path.join(env.PATH_DATA_DIR, config_exp.train)
    path_test = os.path.join(env.PATH_DATA_DIR, config_exp.test)
    paths_pdf = os.path.join(env.PATH_DATA_DIR, 'raw', '건설안전지침')
    paths_pdf = [os.path.join(paths_pdf, x) for x in os.listdir(paths_pdf)]
    path_exp = os.path.join(env.PATH_LOG_DIR, "exp", now)

    # 실험 파라미터
    encoding = config_exp.data_encoding
    pipeline = config_exp.data_pipeline
    prompt_template = config_exp.prompt_template
    chain_type1 = config_exp.RAG_chain_type1
    chain_type2 = config_exp.RAG_chain_type2
    model_name = config_exp.model_name
    temperature = config_exp.temperature
    top_p = config_exp.top_p
    top_k = config_exp.top_k
    max_new_tokens = config_exp.max_new_tokens

    model_params = {
        "model_name":model_name,
        "temperature":temperature,
        "top_p" :top_p,
        "top_k":top_k,
        "max_new_tokens":max_new_tokens
    }

    pipeline = install_pipeline(pipeline)

    # 데이터 로드
    test_data = pd.read_csv(path_test, encoding = encoding)
    precendent = pd.read_csv(path_train, encoding = encoding)
    guidelines = PDFDataset(paths_pdf)

    test_data = pipeline(test_data)
    precendent = pipeline(precendent)

    precendents = []
    for i, row in precendent.iterrows():
        prec = get_prompt_precendent(row)
        precendents.append(prec)
    
    # 벡터스토어 생성
    retriever_precendent = build_vectorstore(precendents)
    # retriever_guidelines = build_vectorstore(guidelines)
    

    # LLM 모델 로드
    llm = load_LLM('load_vllm', model_params)

    # 템플릿 프롬프트
    prompt_template = get_prompt_template(exp = prompt_template)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template.template,
    )
    # RAG 체인 (DF 기반)
    chain_df = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type1,
        retriever=retriever_precendent,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # # RAG 체인 (PDF 기반)
    # chain_pdf = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type=chain_type2,
    #     retriever=retriever_guidelines,
    #     return_source_documents=True,
    #     chain_type_kwargs={"prompt": prompt}
    # )

    # 추론
    test_results = []
    for idx, row in tqdm(test_data.iterrows()):
        question = get_prompt_question(row)
        result_df = chain_df.invoke(question)
        
        # result_pdf = chain_pdf.invoke(question)

        # 사용자가 원하는 방식으로 두 결과를 합치거나, 둘 중 하나만 선택
        # 여기서는 DF 결과와 PDF 결과를 단순 연결 예시
        final_result = result_df['result']
        test_results.append(final_result)
    
    # 저장
    os.makedirs(path_exp, exist_ok=True)
    path_param = os.path.join(path_exp, 'exp.json')
    path_result = os.path.join(path_exp, 'result.csv')
    with open(path_param, 'w') as f:
        json.dump(config_exp.to_dict(), f)
    pd.DataFrame(test_results, columns=['answer']).to_csv(path_result, index=False)





if __name__ == "__main__":
    import torch

    print("CUDA 사용 가능:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU 개수:", torch.cuda.device_count())
        print("사용 중인 GPU:", torch.cuda.get_device_name(0))


    infer_baseline_1()
    