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
from .settings import settings

router = APIRouter()

class MatchRequest(BaseModel):
    query: str
    document: str
    k: int

class MatchedResponse(BaseModel):
    sentences: list[ScoredSentences]

@router.post("/matching", response_model=MatchedResponse)
async def matching(body: MatchRequest):
    embedding_strategy = LLAMA2Embedding(settings.ollama_host)
    distance_strategy = CosineDistance()
    model = MatchingModel(embedding_strategy, distance_strategy)
    sentences = await model.compute(body.query, body.document)
    choices = sentences[:body.k]
    return { "sentences": choices }

class ChatRequest(BaseModel):
    query: str
    sentences: list[str]

@router.post("/chat")
async def chat(body: ChatRequest):
    llm = LLAMA2Service(settings.ollama_host)
    model = ChatModel(llm)
    return EventSourceResponse(model.compute(body.sentences, body.query))
