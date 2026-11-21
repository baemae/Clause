from langchain.tools import tool
from langchain_qdrant import QdrantVectorStore


def file_search(vector_store: QdrantVectorStore, top_k: int):
    @tool(response_format="content_and_artifact")
    def search_handbook(query: str) -> tuple[str, list[dict]]:
        """Search the Compnay handbook for the relevant query. Pass a short, specific query."""
        results = vector_store.similarity_search_with_score(
            query=query, k=top_k
        )
        if not results:
            return "Nothing found", []

        #texts = [doc.page_content for doc, _ in results]
        text = "\n\n".join(doc.page_content for doc, _ in results)
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "score": round(score, 4),
            }
            for doc, score in results
        ]
        return text, sources
    return search_handbook
    