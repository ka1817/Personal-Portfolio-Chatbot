import os
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataSplitting


def test_chunking_from_info_txt():
    ingestor = DataIngestion()
    df = ingestor.load_data()

    splitter = DataSplitting()
    chunks = splitter.chunking()

    assert not df.empty, "personal_portfolio_dataset should not be empty"
    assert len(chunks) > 0, "Text should be converted into at least one Document"
    assert all(hasattr(chunk, "page_content") for chunk in chunks), \
        "Each chunk should have page_content"
