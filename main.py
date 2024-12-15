import mlflow
from config.config import Config
from src.data_processing import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.qa_pipeline import QAPipeline

def main():
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)

    
    with mlflow.start_run():
        
        doc_processor = DocumentProcessor(Config)
        documents = doc_processor.load_and_split_document()

        embedding_manager = EmbeddingManager(Config)
        retriever = embedding_manager.create_vector_store(documents)

        qa_pipeline = QAPipeline(Config, retriever)
        
        
        question = "ce-quoi le role de la loi 01-00"
        sub_questions = qa_pipeline.generate_decomposition_queries(question)

        print("Sub-questions generated:", sub_questions)

        
        decomposition_qa = qa_pipeline.generate_decomposition_qa(sub_questions)
        print("Decomposition QA:", decomposition_qa)

if __name__ == "__main__":
    main()