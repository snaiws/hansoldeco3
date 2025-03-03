import pandas as pd
from typing import Dict

from .base import BasePrecendentDataset



class CSVPrecendentDataset(BasePrecendentDataset):
    """
    CSV 파일에서 chunk_size 단위로 데이터를 순차 로드 (IterableDataset).
    parse_fn, transform_fn을 이용해 후처리를 적용할 수 있다.
    """

    def __init__(self, 
                 csv_path: str, 
                 chunk_size: int = 1024,
                 encoding: str = 'utf-8-sig', 
                 parser=None, 
                 pipeline=None,
                 **read_csv_kwargs):
        super().__init__(
            chunk_size=chunk_size,
            parser=parser,
            pipeline=pipeline
        )
        self.csv_path = csv_path
        self.encoding = encoding
        self.read_csv_kwargs = read_csv_kwargs

        # (선택) CSV 전체 길이를 미리 구해두고 싶으면 아래처럼
        # self.total_length = self._get_total_length()

    def fetch_chunk(self, offset: int, chunk_size: int) -> List[Dict[str, Any]]:
        """
        offset ~ offset+chunk_size-1 행을 한 번에 로드
        (pandas의 skiprows, nrows 활용)

        - CSV가 매우 큰 경우에는 이 방식이 비효율적일 수 있음
          => pyarrow, polars 등을 고려
        """
        if offset == 0:
            # 첫 호출
            df = pd.read_csv(
                self.csv_path,
                encoding=self.encoding,
                nrows=chunk_size,
                **self.read_csv_kwargs
            )
        else:
            skiprows = range(1, offset + 1)  # header=0 번째, offset까지 스킵
            df = pd.read_csv(
                self.csv_path,
                encoding=self.encoding,
                skiprows=skiprows,
                nrows=chunk_size,
                header=0,
                **self.read_csv_kwargs
            )
        if df.empty:
            return []
        return df.to_dict('records')

    # def _get_total_length(self):
    #     df = pd.read_csv(self.csv_path, encoding=self.encoding)
    #     return len(df)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    csv_path = ""

    # dataset 생성
    dataset = CSVPrecendentDataset(
        csv_path=csv_path,
        chunk_size=2000,
        parser=None,
        pipeline=None
    )

    # DataLoader로 감싸기 (batch_size=32 등)
    loader = DataLoader(dataset, batch_size=32)

    for batch_idx, batch in enumerate(loader):
        # batch는 길이가 32인 list of dict (or dict of list) 형태가 될 것
        # parse_fn + transform_fn이 적용된 question/answer가 들어있음
        print(f"Batch {batch_idx}: size={len(batch)}")
        
        # 예시: 첫 레코드
        if len(batch) > 0:
            print(batch[0])
        
        if batch_idx == 2:
            break

