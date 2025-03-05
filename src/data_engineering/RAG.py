from typing import Iterable

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm


def build_vectorstore(
    data: Iterable[str],
    embedding_model_name: str = "jhgan/ko-sbert-nli",
    search_type:str = "similarity",
    search_kwargs:dict = {"k": 5},
):
    """
    1. 여러 개의 string 데이터를 입력받아
    2. 임베딩 후
    3. 벡터스토어에 저장
    4. 리트리버 리턴
    추후 역할을 쪼개고 클래스화 필요
    """


    # 임베딩 생성
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    docs = []
    for batch_idx, batch in tqdm(enumerate(data)):
        docs.append(batch)
    # DF 문서용 VectorStore 생성
    vector_store_df = FAISS.from_texts(docs, embedding)
    retriever_df = vector_store_df.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    
    return retriever_df


