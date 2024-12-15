import os
from dotenv import load_dotenv

load_dotenv("var.env")

class Config:
   
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')

    
    PDF_PATH = "/home/moussa/Desktop/MLFlow_RAG_with _MLFlow/data/loi-oo-o1.pdf"

    
    MLFLOW_TRACKING_URI = "mlruns"
    EXPERIMENT_NAME = "RAG_Document_QA"
