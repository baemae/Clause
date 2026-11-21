#indexing
# Answering questions from a PDF document using LangChain and Qdrant

import time

from langchain.chat_models import init_chat_model
from langchain_google_genai.chat_models import GoogleRateLimitError
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse   # changed
from pydantic import BaseModel, Field

from app.core.config import get_settings, get_embeddings, get_qdrant_client


NO_ANSWER = "No relevant information found. The Clause shop handbook does not cover that topic."

USER_TEMPLATE = (
    "<context>\n{context}\n</context>\n\n"
    "<question>\n{question}\n</question>"
)


class Source(BaseModel):
    source: str = Field(min_length=2)
    page: int | None = None
    score: float


class Answer(BaseModel):
    answer: str
    sources: list[Source]


class RAGService:
    def __init__(self) -> None:
        self.settings = get_settings()  # settings class object / instance
        self.client = get_qdrant_client(self.settings)

        if not self.client.collection_exists(collection_name=self.settings.collection):
            raise RuntimeError(f"Collection {self.settings.collection} does not exist")

        self.vector_store = QdrantVectorStore(                                    # changed
            client=self.client,
            collection_name=self.settings.collection,
            embedding=get_embeddings(self.settings),
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

        # Gemini models ignore temperature (and print a warning), so only pass it
        # for other providers such as Ollama. Check that "google_genai" matches
        # the value of chat_provider in your settings.
        llm_kwargs = {}
        if self.settings.chat_provider != "google_genai":
            llm_kwargs["temperature"] = self.settings.chat_temperature

        self.llm = init_chat_model(
            model=self.settings.chat_model,
            model_provider=self.settings.chat_provider,
            api_key=self.settings.chat_api_key,
            **llm_kwargs,
        )
        self._last_call = 0.0

    def _invoke(self, message) -> str:
        # pace calls: free tier allows 15/min, so keep at least 4.5s between them
        wait = 4.5 - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

        for attempt in range(5):
            try:
                text = self.llm.invoke(input=message).text
                self._last_call = time.monotonic()
                return text
            except GoogleRateLimitError:
                time.sleep(15 * (attempt + 1))  # 15s, 30s, 45s, 60s, 75s
        raise RuntimeError("Still rate limited after retries")

    def ask(self, question: str):
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        results = self.vector_store.similarity_search_with_score(
            query=question,
            k=self.settings.top_k,
            score_threshold=self.settings.score_threshold,
        )

        if not results:
            return Answer(answer=NO_ANSWER, sources=[])  # saves token

        context = "\n\n".join(
            f"[Excerpt {i}]\n{doc.page_content}"
            for i, (doc, _score) in enumerate(results, start=1)
        )

        message = [
            {"role": "system", "content": self.settings.system_prompt},
            {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
        ]
        return Answer(  # gives answer along with the score i.e citation
            answer=self._invoke(message),
            sources=[
                Source(
                    source=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page"),
                    score=round(number=score, ndigits=4),
                )
                for doc, score in results
            ],
        )

    def close(self) -> None:
        self.client.close()