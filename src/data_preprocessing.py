import logging
from langchain.schema import Document
from src.data_ingestion import DataIngestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataSplitting:

    def __init__(self):
        logger.info("Initialized DataSplitting")

    def chunking(self):
        logger.info("Starting document ingestion...")
        data = DataIngestion()
        df = data.load_data()
        logger.info(f"Received data with shape {df.shape} for creating documents.")

        chunks = []
        for i, row in df.iterrows():
            text = f"Question: {row['query']}\nAnswer: {row['response']}"
            chunks.append(Document(page_content=text))
        logger.info(f"Converted {len(chunks)} rows into Document objects.")

        return chunks


