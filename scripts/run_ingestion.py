from src.data_ingestion import DataIngestion

if __name__ == "__main__":
    ingestion = DataIngestion()
    docs = ingestion.load_data()
    print(f"Loaded {len(docs)} documents.")
