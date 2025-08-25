import os
import logging
import warnings
from dotenv import load_dotenv
from src.data_ingestion import DataIngestion
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

logging.getLogger().setLevel(logging.WARNING)


load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant")


class RetrievalGeneration:
    def __init__(self, vectorstore_path: str = "faiss_store"):
        self.vectorstore_path = vectorstore_path
        self.vectorstore = None
        logger.info("RetrievalGeneration initialized with path: %s", vectorstore_path)

    def init_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        logger.info("Embeddings model loaded: sentence-transformers/all-mpnet-base-v2")

        try:
            if os.path.exists(self.vectorstore_path):
                logger.info("Loading existing FAISS index from local storage: %s", self.vectorstore_path)
                self.vectorstore = FAISS.load_local(
                    self.vectorstore_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS index successfully loaded (CPU mode).")
            else:
                logger.warning("No FAISS index found. Building a new one...")
                splits = DataSplitting(chunk_size=1500, chunk_overlap=500)
                chunks = splits.chunking()
                logger.info("Data split into %d chunks", len(chunks))

                self.vectorstore = FAISS.from_documents(chunks, embeddings)
                self.vectorstore.save_local(self.vectorstore_path)
                logger.info("FAISS index built and saved at: %s (CPU mode)", self.vectorstore_path)

        except Exception as e:
            logger.error("Error while initializing FAISS: %s", str(e))
            raise

        return self.vectorstore

    def build_rag_chain(self):
        """Build a custom RAG pipeline with prompt + retriever + LLM."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Run init_vectorstore() first.")

        logger.info("Creating retriever from FAISS vectorstore...")
        custom_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        logger.info("Defining custom RAG prompt template...")
        prompt = PromptTemplate(
            template="""
You are a smart and knowledgeable AI assistant helping users understand 
the professional background, projects, skills, and certifications of Katta Sai Pranav Reddy.

Use the following context extracted from Pranav's profile and provide a clear, helpful, and detailed answer.

Context:
{context}

Question: {question}
Helpful Answer:""",
            input_variables=["context", "question"]
        )

        logger.info("Building RAG chain with retriever + prompt + LLM...")
        rag_chain = (
            RunnableParallel({
                "context": custom_retriever,
                "question": RunnablePassthrough()
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("RAG chain successfully built.")
        return rag_chain
