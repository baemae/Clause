# will make an API here
#Web APIs --> web API+ AI --> AI APIs or RAG APIs or Clause APIs
# combo of 2 things pydantic -> handles data parts and starlette -> handles web parts
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from pydantic import BaseModel, StringConstraints

from app.services.chat import RAGService, Answer
@asynccontextmanager
async def Lifespan(app: FastAPI) -> AsyncGenerator[None, None]: #tells FastAPI how to start and stop the app
    app.state.rag = RAGService()
    yield
    app.state.rag.close()

app = FastAPI( title="Clause AI", lifespan=Lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskQuestion(BaseModel):
    question: Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]



@app.get(path="/") #built-in decorator to define a GET endpoint
def health() -> dict[str, str]:
    return {"status": "ok"}

#http methods: GET, POST, PUT, PATCH, DELETE ---- request <---> response
@app.post(path="/ask", response_model=Answer)
def ask_query(body: AskQuestion, req: Request) -> None:
    return req.app.state.rag.ask(body.question)
