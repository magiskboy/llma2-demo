import asyncio
import time
import json
from fastapi import Body, APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from .models import MatchingModel, ScoredSentences, ChatModel
from .distances import CosineDistance
from .embeddings import LLAMA2Embedding
from .llm import LLAMA2Service

router = APIRouter()

class MatchRequest(BaseModel):
    query: str
    document: str
    k: int

class MatchedResponse(BaseModel):
    sentences: list[ScoredSentences]

@router.post("/matching", response_model=MatchedResponse)
async def matching(body: MatchRequest):
    top_k = body.k
    model = MatchingModel(LLAMA2Embedding, CosineDistance)
    sentences = await model.compute(body.query, body.document)
    choices = sentences[:top_k]
    return { "sentences": choices }

class ChatRequest(BaseModel):
    query: str
    sentences: list[ScoredSentences]

@router.post("/chat")
async def chat(body: ChatRequest):
    llm = LLAMA2Service("http://localhost:11434")
    model = ChatModel(llm)

    async def generator():
        async for text in model.compute(body.sentences, body.query):
            data = {
                "text": text,
                "timestamp": time.time(),
            }
            chunk = json.dumps(data) + "\n"
            yield chunk
    return EventSourceResponse(generator())