from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.retrival_generation import RetrievalGeneration
import uvicorn

class QueryRequest(BaseModel):
    query: str

retriever = RetrievalGeneration(vectorstore_path="local_faiss_index")

@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever.init_vectorstore(rebuild=False)   
    retriever.build_rag_chain(k=10, top_n=5)              
    yield

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/predict")
def predict(request: QueryRequest):
    response = retriever.rag_chain.invoke(request.query)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=4000, reload=True)
