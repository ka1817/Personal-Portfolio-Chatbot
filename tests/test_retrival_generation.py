import os
import pytest
from src.retrival_generation import RetrievalGeneration
import warnings
warnings.filterwarnings('ignore')

def test_init_vectorstore_with_existing_index():
    vectorstore_path = "local_faiss_index"
    assert os.path.exists(vectorstore_path), (
        f"Expected FAISS index at {vectorstore_path}, but not found."
        "Run init_vectorstore(rebuild=True) once before tests."
    )

    rg = RetrievalGeneration(vectorstore_path=vectorstore_path)
    vs = rg.init_vectorstore(rebuild=False)
    assert vs is not None
    assert hasattr(vs, "as_retriever")


def test_build_rag_chain():
    rg = RetrievalGeneration(vectorstore_path="local_faiss_index")
    rg.init_vectorstore(rebuild=False)
    rag_chain = rg.build_rag_chain(k=3, top_n=2)

    assert rag_chain is not None

    result = rag_chain.invoke("What are Pranav Reddy's skills?")
    assert isinstance(result, str)
