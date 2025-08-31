import os
import pytest
from src.data_ingestion import DataIngestion


@pytest.fixture
def sample_text_file(tmp_path):
    """Fixture: creates a temporary text file with sample portfolio data."""
    test_file = tmp_path / "sample_info.txt"
    test_file.write_text("KATTA SAI PRANAV REDDY - Sample Portfolio Data")
    return str(test_file)


def test_default_file_exists():
    """✅ Ensure the default info.txt file path exists."""
    ingestor = DataIngestion()
    assert os.path.exists(ingestor.path), "Default info.txt file should exist"


def test_load_custom_file(sample_text_file):
    """✅ Ensure custom file ingestion works correctly."""
    ingestor = DataIngestion(sample_text_file)
    docs = ingestor.load_data()

    # Verify file is read and contains expected content
    assert len(docs) > 0, "Ingested document list should not be empty"
    assert "PRANAV REDDY" in docs[0].page_content, "Expected text not found in ingested content"


def test_missing_file_raises_error(tmp_path):
    """❌ Ensure missing file raises FileNotFoundError."""
    missing_file = tmp_path / "does_not_exist.txt"
    ingestor = DataIngestion(str(missing_file))

    with pytest.raises(FileNotFoundError):
        ingestor.load_data()
