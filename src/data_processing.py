import mlflow
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
    
    def load_and_split_document(self):
        with mlflow.start_run(nested=True):
            
            mlflow.log_params({
                "document_path": self.config.PDF_PATH,
                "chunk_size": 1000,
                "chunk_overlap": 200
            })

            
            loader = PyMuPDFLoader(self.config.PDF_PATH)
            documents = loader.load()

           
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            
            mlflow.log_metric("num_document_chunks", len(splits))

            return splits
