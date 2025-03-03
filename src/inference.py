from langchain.chains import RetrievalQA

from model.LLM import load_llm_model
from data_engineering.RAG import build_vectorstore
from data_engineering.dataset.precendent import CSVPrecendentDataset
from data_engineering.dataset.guideline import PDFDataset

def infer_baseline():
    """
    (1) 선례 데이터 전처리 후 선례 docs 생성, 선례 데이터 
    (2) DF-기반 Retrievr와 PDF-기반 Retriever 중 선택(또는 병합) 가능
    (3) 결과를 CSV로 저장
    """
    print("5. 추론 & 결과 처리를 시작합니다.")
    
    # configs
    basic_configs = {}
    csv_path = 
    pdf_files = 
    chunk_size = 
    encoding = 'utf-8-sig'
    bnb_config = {}

    chain_type1 = "stuff"
    chain_type2 = "stuff"
    model_name = 
    prompt_template = ""
    pipeline = 

    # 데이터 로드
    test_data = CSVPrecendentDataset(
                 csv_path = csv_path, 
                 chunk_size = chunk_size,
                 encoding = encoding, 
                 parser = None, 
                 pipeline = pipeline,
                 )
    precendent = CSVPrecendentDataset(
                 csv_path = csv_path, 
                 chunk_size = chunk_size,
                 encoding = encoding, 
                 parser = None, 
                 pipeline = pipeline,
                 )
    questions = CSVPrecendentDataset(
                 csv_path = csv_path, 
                 chunk_size = chunk_size,
                 encoding = encoding, 
                 parser = None, 
                 pipeline = pipeline,
                 )
    guidelines = PDFDataset(pdf_files)

    
    # 벡터스토어 생성
    retriever_precendent = build_vectorstore(precendent)
    retriever_guidelines = build_vectorstore(guidelines)

    # LLM 모델 로드
    llm = load_llm_model(model_name, bnb_config)

    # RAG 체인 (DF 기반)
    chain_df = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type1,
        retriever=retriever_precendent,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    # RAG 체인 (PDF 기반)
    chain_pdf = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type2,
        retriever=retriever_guidelines,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    # 추론
    test_results = []
    for idx, row in test_data.iterrows():
        question = row["question"]
        result_df = chain_df.run(question)
        result_pdf = chain_pdf.run(question)

        # 사용자가 원하는 방식으로 두 결과를 합치거나, 둘 중 하나만 선택
        # 여기서는 DF 결과와 PDF 결과를 단순 연결 예시
        final_result = f"[DF 기반 답변]\n{result_df}\n\n[PDF 기반 답변]\n{result_pdf}"
        test_results.append(final_result)

        if (idx + 1) % 10 == 0:
            print(f"   진행 상황: {idx + 1}/{len(test_data)} 완료...")




##############################################
# 메인 실행 (예시)
##############################################
if __name__ == "__main__":
    import torch

    print("CUDA 사용 가능:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU 개수:", torch.cuda.device_count())
        print("사용 중인 GPU:", torch.cuda.get_device_name(0))


    infer_baseline()