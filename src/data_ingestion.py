import os
import logging
from langchain_community.document_loaders import TextLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataIngestion:

    def __init__(self, path: str | None = None):
        if path is None:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "info.txt"
            )
            self.path = os.path.abspath(default_path)
            logger.info(f"No path provided. Using default file: {self.path}")
        else:
            self.path = os.path.abspath(path)
            logger.info(f"Using custom file path: {self.path}")

    def load_data(self):
        logger.debug(f"Checking if file exists at: {self.path}")
        if not os.path.exists(self.path):
            logger.error(f"File not found at {self.path}")
            raise FileNotFoundError(f"File not found: {self.path}")

        logger.info(f"Loading file: {self.path}")
        loader = TextLoader(self.path, encoding="utf-8")
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from {self.path}")
        return docs



