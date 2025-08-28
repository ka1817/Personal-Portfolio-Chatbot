import os
import logging
import warnings
from dotenv import load_dotenv
from src.data_preprocessing import DataSplitting
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


logger = logging.getLogger("RetrievalGeneration")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))


class RetrievalGeneration:
    def __init__(self, vectorstore_path: str = "faiss_store"):
        self.vectorstore_path = vectorstore_path
        self.vectorstore = None
        self.rag_chain = None
        logger.info("RetrievalGeneration initialized with path: %s", vectorstore_path)

    def init_vectorstore(self, rebuild: bool = False):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        logger.info("Embeddings model loaded.")

        if os.path.exists(self.vectorstore_path) and not rebuild:
            logger.info("Loading existing FAISS index from: %s", self.vectorstore_path)
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path, embeddings, allow_dangerous_deserialization=True
            )
        else:
            logger.warning("Building new FAISS index...")
            chunks = DataSplitting(chunk_size=1000, chunk_overlap=200).chunking()
            logger.info("Data split into %d chunks", len(chunks))
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            self.vectorstore.save_local(self.vectorstore_path)
            logger.info("FAISS index saved at: %s", self.vectorstore_path)

        return self.vectorstore

    def build_rag_chain(self, k: int = 5):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Run init_vectorstore() first.")

        logger.info("Creating retriever from FAISS vectorstore (top_k=%d)...", k)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        prompt = PromptTemplate(
            template="""
You are a knowledgeable AI assistant answering questions about
the professional background, projects, skills, and certifications of Katta Sai Pranav Reddy.

If the context does not contain enough information, politely say so.

Context:
{context}

Question: {question}
Helpful Answer:""",
            input_variables=["context", "question"]
        )

        self.rag_chain = (
            RunnableParallel({
                "context": retriever,
                "question": RunnablePassthrough()
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("RAG chain successfully built.")
        return self.rag_chain

    