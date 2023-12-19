from typing import AsyncIterator
from ._base import Computation
from .matching import ScoredSentences
from ..llm import LLMService

class ChatModel(Computation):
    def __init__(self, llm: LLMService):
        self.llm = llm

    async def compute(self, sentences: list[str], query: str) -> AsyncIterator[str]:
        join_context = ".\n".join(sentences)
        prompt = f"""
            You are giving the context and the question, try to generate the answer with the given context:
            context: {join_context}.
            question: {query}."""

        async for text in self.llm.completion(prompt):
            yield text
