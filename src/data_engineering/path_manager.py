import os

class PathManager:
    """
    - 환경 변수 혹은 직접 지정한 베이스 디렉터리를 기반으로
      Data, Model, Log 등 여러 디렉터리 경로를 손쉽게 관리하는 클래스.
    - 필요하다면 디렉터리 구조를 자유롭게 확장하고,
      경로 가져오기/디렉터리 생성 등 기능을 수행할 수 있음.
    """
    
    def __init__(self, base_dir: str = None):
        """
        :param base_dir: 베이스 디렉터리. 없으면 환경변수 'BASE_DATA_DIR'에서 가져옴.
        """
        if base_dir is not None:
            self.base_dir = base_dir
        else:
            # 예: BASE_DATA_DIR 환경변수 사용
            self.base_dir = os.environ.get("BASE_DATA_DIR", ".")
        
        # 디렉터리 구조: 기본값
        # key = 논리적 범주, value = 실제 디렉터리명
        self._structure = {
            "Data": "data",
            "Model": "model",
            "Log": "log"
        }
    
    def set_structure(self, structure: dict):
        """
        디렉터리 구조를 바꾸고 싶을 때 사용.
        
        :param structure: 예) {"Data": "my_data_dir", "Model": "my_model_dir", "Log": "my_log_dir"}
        """
        self._structure = structure
    
    def get_path(self, category: str, *subpaths) -> str:
        """
        주어진 범주(category)의 디렉터리 경로에 추가적인 하위 경로(subpaths)를
        붙여서 최종 경로를 반환.
        
        :param category: "Data", "Model", "Log" 등
        :param subpaths: "train", "images" 등 하위 디렉터리/파일 경로
        :return: 최종적인 파일/디렉터리 경로
        """
        if category not in self._structure:
            raise ValueError(f"정의되지 않은 category입니다: {category}")
        
        cat_path = self._structure[category]
        full_path = os.path.join(self.base_dir, cat_path, *subpaths)
        return full_path
    
    def ensure_dir_exists(self, category: str, *subpaths) -> str:
        """
        해당 범주의 경로(하위 경로 포함)가 실제로 존재하지 않으면 디렉터리를 생성.
        
        :param category: "Data", "Model", "Log" 등
        :param subpaths: 하위 디렉터리 경로
        :return: 최종적인 디렉터리 경로
        """
        dir_path = self.get_path(category, *subpaths)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path


if __name__ == "__main__":
    # 예시 사용
    pm = PathManager()  # base_dir이 없으면 환경변수 'BASE_DATA_DIR'를 사용

    # 구조 변경(원한다면)
    # pm.set_structure({
    #     "Data": "my_data_dir",
    #     "Model": "my_model_dir",
    #     "Log": "my_log_dir"
    # })

    # 경로 가져오기
    data_path = pm.get_path("Data")  # base_dir/data
    print("Data Path:", data_path)

    # 하위 폴더까지 포함된 경로
    train_data_path = pm.get_path("Data", "train", "images")
    print("Train Images Path:", train_data_path)

    # 디렉터리 생성
    pm.ensure_dir_exists("Log", "2025", "03", "03")
    print("Log 디렉터리 생성 완료")
