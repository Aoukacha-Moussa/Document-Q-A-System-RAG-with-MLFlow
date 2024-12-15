import mlflow
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter 

class QAPipeline:
    def __init__(self, config, retriever):
        self.config = config
        self.retriever = retriever
        self.llm = ChatGoogleGenerativeAI(
            temperature=1, 
            model="gemini-pro", 
            google_api_key=self.config.GOOGLE_API_KEY
        )
    
    def generate_decomposition_queries(self, question):
        with mlflow.start_run(nested=True):
            
            template = """Vous êtes un assistant utile qui génère plusieurs sous-questions liées à une question d'entrée.
            L'objectif est de décomposer la question d'entrée en une série de sous-problèmes ou de sous-questions pouvant être traitées isolément.
            Générez plusieurs requêtes de recherche liées à : {question}
            Résultat (3 requêtes) :"""
            
            prompt_decomposition = ChatPromptTemplate.from_template(template)

           
            generate_queries_decomposition = (
                prompt_decomposition 
                | self.llm 
                | StrOutputParser() 
                | (lambda x: x.split("\n"))
            )

            
            mlflow.log_param("input_question", question)

            
            questions = generate_queries_decomposition.invoke({"question": question})
            
            mlflow.log_metric("num_subquestions", len(questions))

            return questions
    def generate_decomposition_qa(self, questions):
        """
        Generate question-answer pairs using a decomposition prompt.

        Args:
            questions (list): A list of questions to process.

        Returns:
            str: A formatted string of question-answer pairs.
        """
        with mlflow.start_run(nested=True):
            
            template = """Voici la question à laquelle vous devez répondre:

            \n --- \n {question} \n --- \n
            Voici les paires question-réponse disponibles comme référence:

            \n --- \n {q_a_pairs} \n --- \n
            Voici le contexte supplémentaire pertinent pour la question:

            \n --- \n {context} \n --- \n
            Utilisez le contexte ci-dessus et les paires question-réponse pour répondre à la question: \n {question}

            Instructions supplémentaires:
            - Répondez en français
            - Utilisez un langage clair et précis
            - Basez votre réponse uniquement sur les informations fournies
            - Assurez-vous que votre réponse est pertinente et précise
            - Assurez-vous que votre réponse est bien structurée et cohérente
            - Assurez-vous que votre réponse est bien argumentée et expliquée
            - Assurez-vous que votre réponse les sources et les références appropriées 

            """

            decomposition_prompt = ChatPromptTemplate.from_template(template)

            def format_qa_pair(question, answer):
                """Format Q and A pair"""
                return f"Question: {question}\nAnswer: {answer}\n"

            
            q_a_pairs = ""

            
            for q in questions:
                rag_chain = (
                    {"context": itemgetter("question") | self.retriever,
                    "question": itemgetter("question"),
                    "q_a_pairs": itemgetter("q_a_pairs")}
                    | decomposition_prompt
                    | self.llm
                    | StrOutputParser()
                )

               
                answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
                q_a_pair = format_qa_pair(q, answer)
                q_a_pairs += f"\n---\n{q_a_pair}"

            
            mlflow.log_param("num_questions", len(questions))
            mlflow.log_metric("q_a_pairs_length", len(q_a_pairs))

            return q_a_pairs.strip()
