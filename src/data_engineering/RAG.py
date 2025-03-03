
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from torch.utils.data import DataLoader



def build_vectorstore(
    dataloader: DataLoader,
    embedding_model_name: str = "jhgan/ko-sbert-nli",
    search_type:str = "similarity",
    search_kwargs:dict = {"k": 5},
):
    """
    (A) DataFrame의 훈련용 QA 데이터를 Text 형태로 변환 후
        FAISS 벡터스토어를 구축.
    """
    print("3-A. DataFrame 기반 벡터스토어를 생성합니다.")


    # 임베딩 생성
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

    docs = []
    for batch_idx, batch in enumerate(dataloader):
        for doc in batch:
            docs.append(doc)

    # DF 문서용 VectorStore 생성
    vector_store_df = FAISS.from_texts(docs, embedding)
    retriever_df = vector_store_df.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    
    print(" - DF 기반 벡터스토어 & 리트리버 생성 완료!")
    return retriever_df


