from typing import AsyncIterator
from ._base import Computation
from .matching import ScoredSentences
from ..llm import LLMService

class ChatModel(Computation):
    def __init__(self, llm: LLMService):
        self.llm = llm

    async def compute(self, sentences: list[ScoredSentences], query: str) -> AsyncIterator[str]:
        messages = []
        for sentence in sentences:
            messages.append({
                "content": sentence.text,
                "role": "user",
            })
        messages.append({
            "content": query,
            "role": "user",
        })

        async for text in self.llm.chat(messages):
            yield text