"""
FastAPI server wrapping the triage agent.

Exposes POST /chat that LiteMerge-UI can call as an agent endpoint.

Request body:
    { "message": str, "session_id": str, "conversation_history": [...] }

Response:
    { "reply": str }

Run with:
    python server.py
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from triage_agent import triage_to_str

app = FastAPI(title="Triage Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""
    conversation_history: list = []


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    reply = triage_to_str(request.message)
    return ChatResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
