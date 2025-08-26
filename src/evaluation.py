import os
from dotenv import load_dotenv
from datasets import Dataset

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.run_config import RunConfig

from src.retrival_generation import RetrievalGeneration


class Evaluation:
    def __init__(self, vectorstore_path: str, llm_model: str = "llama-3.3-70b-versatile"):
        # Load environment and API keys
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # Initialize LLM
        self.llm = ChatGroq(api_key=self.groq_api_key, model=llm_model)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Initialize vectorstore
        self.vectorstore_path = vectorstore_path
        self.vectorstore = FAISS.load_local(
            self.vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Initialize retrieval-generation pipeline
        self.rg = RetrievalGeneration(vectorstore_path=vectorstore_path)
        self.rg.init_vectorstore()
        self.qa = self.rg.build_rag_chain()

    def run(self, questions: list, ground_truth: list, use_reranker: bool = False):
        """Run evaluation with or without reranking"""

        if use_reranker:
            # Load reranker
            cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=3)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.vectorstore.as_retriever(search_kwargs={"k": 10})
            )
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

        answers, contexts = [], []
        for query in questions:
            answers.append(self.qa.invoke(query))
            contexts.append([doc.page_content for doc in retriever.get_relevant_documents(query)])

        data = {
            "question": questions,
            "ground_truth": ground_truth,
            "answer": answers,
            "contexts": contexts
        }

        dataset = Dataset.from_dict(data)

        # Configure Ragas evaluation
        run_config = RunConfig(
            timeout=290,
            max_retries=5,
            max_wait=30,
            max_workers=1
        )

        result = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            llm=self.llm,
            embeddings=self.embeddings,
            run_config=run_config,
            batch_size=1
        )

        return result
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    vectorstore_path = os.path.join(base_dir, "..", "local_faiss_index")

    evaluation = Evaluation(vectorstore_path)

    questions = [
        "What were Katta Sai Pranav Reddy’s 10th class marks and CGPA?",
        "What subjects did Pranav Reddy study in 12th (Intermediate) and what were his marks?",
        "Can you summarize Pranav Reddy’s professional and project experience?"
    ]

    ground_truth = [
        "Katta Sai Pranav Reddy completed his SSC in March 2019 at Ekalavya Foundation School, Nalgonda, securing A1 grades in most subjects and a B1 in Hindi, with an overall CGPA of 9.5.",
        "In March 2021, Pranav Reddy finished his Intermediate education, achieving nearly full marks in English, Sanskrit, HE, and optional subjects like Mathematics, Physics, and Chemistry, with a total of 982 marks.",
        "Pranav Reddy is an AI and ML engineer with internship experience at iNeuron Intelligence and Unified Mentor, where he worked on customer segmentation and attrition prediction. His projects include the BigBasket SmartCart AI Assistant and Netflix Churn Prediction, showcasing skills in Python, ML pipelines, FAISS, FastAPI, and Generative AI solutions."
    ]

    # Run without reranker
    print("🔹 Baseline Evaluation (no reranker)")
    baseline_result = evaluation.run(questions, ground_truth, use_reranker=False)
    print(baseline_result)

    # Run with reranker
    print("\n🔹 Evaluation with Reranker")
    rerank_result = evaluation.run(questions, ground_truth, use_reranker=True)
    print(rerank_result)
