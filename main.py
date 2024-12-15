import mlflow
from config.config import Config
from src.data_processing import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.qa_pipeline import QAPipeline

def process_question(question):
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("input_question", question)
        
        doc_processor = DocumentProcessor(Config)
        documents = doc_processor.load_and_split_document()

        embedding_manager = EmbeddingManager(Config)
        retriever = embedding_manager.create_vector_store(documents)

        qa_pipeline = QAPipeline(Config, retriever)
        
        sub_questions = qa_pipeline.generate_decomposition_queries(question)
        mlflow.log_metric("num_subquestions", len(sub_questions))

        decomposition_qa = qa_pipeline.generate_decomposition_qa(sub_questions)
        mlflow.log_metric("q_a_pairs_length", len(decomposition_qa))

        return decomposition_qa

if __name__ == "__main__":
    #question = "Quels sont les objectifs principaux de l'enseignement supérieur selon cette loi ?"
    #question = "Quels sont les rôles du conseil de l’université selon l’article 12 ?"
    #question = "How does this law ensure equality in access to higher education?"
    question = "What penalties are described for unauthorized private higher education institutions?"
    response = process_question(question)
    print(response)