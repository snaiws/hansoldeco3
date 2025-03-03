import abc
from typing import List, Any, Dict, Callable

from torch.utils.data import IterableDataset



class BasePrecendentDataset(IterableDataset, metaclass=abc.ABCMeta):
    """
    이전 사례 데이터 모델의 기본 클래스
    """
    def __init__(
        self,
        chunk_size: int = 1024,
        parser: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]] = None,
        pipeline = None
    ):
        """
        chunk_size: 한 번에 가져올 레코드 수
        parse_fn:   chunk 데이터를 "파싱"하는 함수 (옵션)
        transform_fn: chunk 데이터를 "추가 변환"하는 함수 (옵션)
          - parse_fn -> transform_fn 순서대로 적용
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.parser = parser
        self.pipeline = pipeline


    @abc.abstractmethod
    def get_source(self, offset: int, chunk_size: int) -> List[Any]:
        """
        offset부터 chunk_size 만큼 데이터를 로드해
        record(또는 row) 목록(List)를 반환한다.
        반환된 List가 비어 있으면 더 이상 데이터가 없다고 판단.
        """
        pass

    def parse(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        데이터 모델 고려한 메소드
        chunk 단위로 "파싱"을 진행하는 훅.
        parse_fn이 주어지면 parse_fn(records)를 호출,
        없으면 그대로 반환.
        """
        if self.parser is not None:
            return self.parser(records)
        return records
    
    def process(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        chunk 단위로 "추가 변환"을 진행하는 훅.
        pipeline 주어지면 process(records)를 호출,
        없으면 그대로 반환.
        """
        if self.pipeline is not None:
            return self.pipeline(records)
        return records


    def __iter__(self):
        offset = 0
        while True:
            records = self.fetch_chunk(offset, self.chunk_size)
            if not records:
                # 더 이상 데이터가 없으면 종료
                break

            # 1) parse
            records = self.parse(records)
            # 2) process
            records = self.process(records)

            # 3) chunk 내 레코드들을 하나씩 yield
            for r in records:
                yield r

            offset += self.chunk_size
