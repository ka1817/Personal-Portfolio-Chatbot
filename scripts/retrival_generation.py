from src.retrival_generation import RetrievalGeneration
rg = RetrievalGeneration(vectorstore_path="local_faiss_index")

rg.init_vectorstore()

qa = rg.build_rag_chain()
answer = qa.invoke("give me Pranav reddy projects Details")
print(answer)
