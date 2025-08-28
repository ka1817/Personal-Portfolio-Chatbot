import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.retrival_generation import RetrievalGeneration


def main():

    retriever = RetrievalGeneration(vectorstore_path="faiss_store")

    retriever.init_vectorstore(rebuild=False)

    rag_chain = retriever.build_rag_chain(k=5)

    
    query = "Give me the contact details of pranav reddy" 

    print(f"\n❓ Question: {query}")
    response = rag_chain.invoke(query)
    print(f"🤖 Answer: {response}")


if __name__ == "__main__":
    main()
