class Embedding:
    async def compute(self, sentence: str) -> list[float]:
        raise NotImplementedError()
