import os
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataSplitting


def test_chunking_from_info_txt():

    ingestor = DataIngestion()
    docs = ingestor.load_data()

    splitter = DataSplitting(chunk_size=150, chunk_overlap=30)
    chunks = splitter.chunking()

    assert os.path.exists(ingestor.path), "info.txt file should exist"
    assert len(docs) > 0, "info.txt should not be empty"
    assert len(chunks) > 1, "Text should be split into multiple chunks"
    assert all(hasattr(chunk, "page_content") for chunk in chunks), \
        "Each chunk should have page_content"
    assert "Pranav" in " ".join(c.page_content for c in chunks), \
        "Chunks should include expected name from info.txt"
