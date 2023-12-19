from typing import AsyncIterator, Protocol

class LLMService(Protocol):
    async def completion(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        raise NotImplementedError()

    async def chat(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        raise NotImplementedError()

    async def calculate_embedding(self, prompt: str) -> list[float]:
        raise NotImplementedError()
