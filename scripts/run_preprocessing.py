from src.data_preprocessing import DataSplitting
def main():
    splitter = DataSplitting()
    chunks = splitter.chunking()
    print(f"Created {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
