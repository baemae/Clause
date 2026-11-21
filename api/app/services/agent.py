from dataclasses import dataclass
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from app.core.config import get_embeddings
from app.services.chat import RAGService
from app.prompts.loaders import get_prompt
# for short-term memory
from langgraph.checkpoint.memory import InMemorySaver

# for long-term memory
from langgraph.store.memory import InMemoryStore



@dataclass
class Context:
    """Per-request info passed to the agent. user_id decides whose memories are used."""
    user_id: str


@tool
def save_memory(fact: str, runtime: ToolRuntime[Context]) -> str:
    """Save one short, lasting fact about the customer (name, preferences, ongoing issue)."""
    runtime.store.put(
        ("memories", runtime.context.user_id), str(uuid4()), {"text": fact}
    )
    return "Saved."


@dynamic_prompt
def prompt_with_memories(request: ModelRequest) -> str:
    """Before every model call, look up this customer's saved notes and add them to the prompt."""
    store = request.runtime.store
    user_id = request.runtime.context.user_id
    latest_question = next(
        (m.content for m in reversed(request.messages) if isinstance(m, HumanMessage)),
        "",
    )
    items = store.search(("memories", user_id), query=str(latest_question), limit=3)
    system_prompt = request.system_prompt
    if not items:
        return system_prompt
    notes = "\n".join(f"- {item.value['text']}" for item in items)
    return f"{system_prompt}\n\nSaved notes about this customer:\n{notes}"


def _to_text(content, strip: bool = True) -> str:
    """Gemini returns a list of content blocks; other models return a plain string.
    Pass strip=False for streamed chunks, or the spaces between words are lost."""
    if isinstance(content, str):
        text = content
    else:
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        text = "".join(parts)
    return text.strip() if strip else text


class ClassAgent:
    def __init__(self) -> None:
        self.rag = RAGService()  # composition

        # Create this once. If you build a new ClassAgent per request,
        # the memory resets every time. Switch to SqliteSaver/PostgresSaver
        # when you need memory to survive restarts.
        self.checkpointer = InMemorySaver()

        # Long-term memory: lives across threads, separated per user.
        # Semantic search needs embeddings; dims is measured, not guessed.
        embeddings = get_embeddings(self.rag.settings)
        self.store = InMemoryStore(
            index={
                "embed": embeddings,
                "dims": len(embeddings.embed_query("test")),
            }
        )

        @tool(response_format="content_and_artifact")
        def search_handbook(query: str) -> tuple[str, list[dict]]:
            """Search the Clause Shop handbook (orders, refunds, shipping, warranty, policies). Pass a short, specific query."""
            results = self.rag.vector_store.similarity_search_with_score(
                query=query, k=self.rag.settings.top_k
            )
            if not results:
                return "Nothing found", []

            texts = [doc.page_content for doc, _ in results]
            sources = [
                {"page": doc.metadata.get("page"), "score": round(score, 4)}
                for doc, score in results
            ]
            return "\n\n".join(texts), sources

        self.agent = create_agent(
            model=self.rag.llm,
            tools=[search_handbook, save_memory],
            middleware=[prompt_with_memories],  # builds the system prompt (SYSTEM + saved notes)
            checkpointer=self.checkpointer,      # short-term memory (per thread)
            store=self.store,                    # long-term memory (per user)
            context_schema=Context,
            system_prompt=get_prompt(
                name="support/system",
                version=self.rag.settings.system_prompt_version),
        )

    def ask(self, question: str, thread_id: str, user_id: str) -> dict:
        """
        Ask a question inside a conversation.

        thread_id: one conversation (short-term memory). Build it from the user,
                   e.g. f"{user_id}:{session_id}", so customers can't load each
                   other's conversations.
        user_id:   the customer (long-term memory, shared across their conversations).
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            context=Context(user_id=user_id),
        )
        return self._read_turn(result["messages"])

    def stream(self, question: str, thread_id: str, user_id: str):
        """
        Same as ask, but yields events as the answer is written.

        Yields dicts:
          {"event": "token", "data": "<piece of the answer>"}
          {"event": "done",  "data": {"sources": [...], "searches": [...]}}
        """
        config = {"configurable": {"thread_id": thread_id}}
        for chunk, meta in self.agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            context=Context(user_id=user_id),
            stream_mode="messages",
        ):
            # Only the model's own words. Skips tool results and other nodes.
            if meta.get("langgraph_node") != "model" or not isinstance(chunk, AIMessage):
                continue
            text = _to_text(chunk.content, strip=False)
            if text:
                yield {"event": "token", "data": text}

        # The checkpointer has the finished turn, so sources come from there.
        messages = self.agent.get_state(config).values["messages"]
        turn = self._read_turn(messages)
        yield {
            "event": "done",
            "data": {"sources": turn["sources"], "searches": turn["searches"]},
        }

    @staticmethod
    def _read_turn(messages: list) -> dict:
        """Answer, sources and searches for the latest turn only."""
        # The checkpointer returns the whole thread, so only look at
        # the messages added during this turn.
        last_human = max(
            i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
        )
        this_turn = messages[last_human + 1:]

        searches = [
            call
            for m in this_turn
            if isinstance(m, AIMessage)
            for call in m.tool_calls
            if call["name"] == "search_handbook"
        ]
        sources = [
            s
            for m in this_turn
            if isinstance(m, ToolMessage) and m.artifact
            for s in m.artifact
        ]
        return {
            "answer": _to_text(messages[-1].content),
            "sources": sources,
            "searches": searches,
        }

    def close(self) -> None:
        self.rag.close()