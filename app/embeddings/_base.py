from typing import Protocol

class Embedding(Protocol):
    async def compute(self, sentence: str) -> list[float]:
        raise NotImplementedError()
