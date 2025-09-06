from src.data_ingestion import DataIngestion

if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.load_data()
    print(f"Loaded {df.shape} documents.")
