import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.retrival_generation import RetrievalGeneration


def main():

    retriever = RetrievalGeneration(vectorstore_path="local_faiss_index")

    retriever.init_vectorstore(rebuild=True)

    rag_chain = retriever.build_rag_chain(k=10,top_n=5)

    
    query = "share pranav reddy skills" 

    print(f"\n❓ Question: {query}")
    response = rag_chain.invoke(query)
    print(f"🤖 Answer: {response}")


if __name__ == "__main__":
    main()
