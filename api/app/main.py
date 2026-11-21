
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.agent import ClassAgent
from app.chat_routes import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:

    app.state.agent = ClassAgent()

    yield

    app.state.agent.close()


app = FastAPI(
    title="Clause Shop AI",
    description="Customer Support Agent with RAG and Memory",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router)


@app.get("/")
def health():
    return {"status": "ok"}
