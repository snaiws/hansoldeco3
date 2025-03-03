from torch.utils.data import Dataset
from langchain.document_loaders import PyPDFLoader

class PDFDataset(Dataset):
    def __init__(self, pdf_paths):
        """
        Args:
            pdf_paths (list): PDF 파일 경로 리스트
        """
        self.pdf_paths = pdf_paths

    def __len__(self):
        return len(self.pdf_paths)

    def __getitem__(self, idx):
        pdf_path = self.pdf_paths[idx]
        text = self.load_pdf(pdf_path)
        return text, pdf_path

    def load_pdf(self, pdf_path):
        """PDF 파일을 로드하여 텍스트를 추출"""
        loader = PyPDFLoader(pdf_path)
        doc = loader.load()  # 각 페이지별 Document
        return doc.page_content
