from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.retrival_generation import RetrievalGeneration

class QueryRequest(BaseModel):
    query: str

retriever = RetrievalGeneration(vectorstore_path="local_faiss_index")

@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever.init_vectorstore(rebuild=False)   
    retriever.build_rag_chain(k=5)              
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "RAG system is running 🚀"}

@app.post("/predict")
def predict(request: QueryRequest):
    response = retriever.rag_chain.invoke(request.query)
    return {"response": response}
