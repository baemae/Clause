
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse 
from app.core.config import Settings,get_embeddings,get_qdrant_client

def get_vector_store(settings: Settings) -> QdrantVectorStore:
    client = get_qdrant_client(settings)
    if not client.collection_exists(collection_name=settings.collection):
        raise RuntimeError(f"Collection {settings.collection} does not exist")

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection,
        embedding=get_embeddings(settings),
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
