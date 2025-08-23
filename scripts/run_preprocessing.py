from src.data_preprocessing import DataSplitting
def main():
    splitter = DataSplitting(chunk_size=50, chunk_overlap=20)
    chunks = splitter.chunking()
    print(f"Created {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
