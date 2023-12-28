from ._base import Embedding
from ..llm import LLAMA2Service

class LLAMA2Embedding(Embedding):
    def __init__(self, base_url = "http://localhost:11434"):
        self.lama2 = LLAMA2Service(base_url)

    async def compute(self, sentence: str) -> list[float]:
        return await self.lama2.calculate_embedding(sentence)
