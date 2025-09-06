import os
import pytest
from src.data_ingestion import DataIngestion
import pandas
@pytest.fixture
def test_load_custom_file(sample_text_file):
    """✅ Ensure custom file ingestion works correctly."""
    ingestor = DataIngestion(sample_text_file)
    df = ingestor.load_data()

    assert not df.empty, "Ingested DataFrame should not be empty"
def test_missing_file_raises_error(tmp_path):
    """❌ Ensure missing file raises FileNotFoundError."""
    missing_file = tmp_path / "does_not_exist.txt"
    ingestor = DataIngestion(str(missing_file))

    with pytest.raises(FileNotFoundError):
        ingestor.load_data()
