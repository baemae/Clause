import re
#from langchain_core.documents import Document
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client.models import Distance, VectorParams, SparseVectorParams


from app.core.config import get_settings, get_embeddings, get_qdrant_client

STAFF_ONLY = re.compile(r"\n?\d+\.[^\n]*\(staff only\).*?(?=\n\d+\. |\Z)", re.S)

# def load(path: Path) -> list[Document]:
#     return PyPDFLoader(path).load()

def load(path: Path) -> list[Document]:
    docs = PyPDFLoader(path).load()
    for doc in docs:
        doc.page_content = STAFF_ONLY.sub("", doc.page_content)
    return docs
      
    

def ingest():
    settings = get_settings()

    #load
    file = load(settings.data_path)

    #2. chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(file)

    #embeddings
    embed = get_embeddings(settings)

    #4. store
    client = get_qdrant_client(settings)

    try:
        if client.collection_exists(collection_name=settings.collection):
            client.delete_collection(collection_name=settings.collection)  # prev table should be deleted before creating new one so that chunks do not duplicate
        dense_vector_params = VectorParams(
            size=settings.embed_dim,
            distance=Distance.COSINE,
        )
        client.create_collection(
            collection_name=settings.collection,
            vectors_config={"dense": VectorParams(size=settings.embed_dim, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams}
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection,
            embedding=embed,
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bn25"),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse"
        )
        vector_store.add_documents(chunks)
    finally:
        client.close()

    return len(chunks)

if __name__ == "__main__":
        print(ingest(), "chunks indexed")