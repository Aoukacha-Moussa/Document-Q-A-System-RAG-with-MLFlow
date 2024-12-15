import mlflow
from langchain.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Chroma

class EmbeddingManager:
    def __init__(self, config):
        self.config = config
    
    def create_vector_store(self, documents):
        with mlflow.start_run(nested=True):
            
            mlflow.log_params({
                "embedding_model": "embed-multilingual-v3.0",
                "vector_store": "Chroma"
            })

            
            embeddings = CohereEmbeddings(
                cohere_api_key=self.config.COHERE_API_KEY,
                model="embed-multilingual-v3.0",
                user_agent="rag-document-qa"
            )

            
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory="./chroma_db12"
            )

            return vectorstore.as_retriever()
