import json
#from functools import lru_cache

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.agent import ClassAgent

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # In production, take user_id from your login/auth, not from the request body.
    user_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


def get_agent(request: Request) -> ClassAgent:
    return request.app.state.agent


def _thread_id(body: ChatRequest) -> str:
    # user + session, so customers can never load each other's conversations
    return f"{body.user_id}:{body.session_id}"


def _sse(event: str, data) -> str:
    """Format one Server-Sent Event. JSON keeps newlines in the text safe."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------- static: wait for the full answer ----------
@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest, agent: ClassAgent = Depends(get_agent)) -> ChatResponse:
    result = agent.ask(
        question=body.question,
        thread_id=_thread_id(body),
        user_id=body.user_id,
    )
    return ChatResponse(answer=result["answer"], sources=result["sources"])


# ---------- streaming: answer arrives piece by piece ----------
@router.post("/stream")
def chat_stream(body: ChatRequest, agent: ClassAgent = Depends(get_agent)) -> StreamingResponse:
    def events():
        try:
            for item in agent.stream(
                question=body.question,
                thread_id=_thread_id(body),
                user_id=body.user_id,
            ):
                if item["event"] == "done":
                    yield _sse("sources", item["data"]["sources"])
                    yield _sse("done", "")
                else:
                    yield _sse("token", item["data"])
        except Exception:
            # Don't leak internal error details to the client
            yield _sse("error", "Something went wrong. Please try again.")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stops nginx from buffering the stream
        },
    )