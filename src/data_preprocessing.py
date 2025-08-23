import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.data_ingestion import DataIngestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataSplitting:

    def __init__(self, chunk_size: int = 40, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            f"Initialized DataSplitting with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    def chunking(self):
        logger.info("Starting document ingestion before splitting...")
        data = DataIngestion()
        docs = data.load_data()
        logger.info(f"Received {len(docs)} documents for splitting.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        logger.debug("Splitter initialized. Splitting documents...")
        chunks = splitter.split_documents(docs)

        logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents.")
        return chunks


