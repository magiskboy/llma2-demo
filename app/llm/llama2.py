from typing import AsyncIterator
import json
import httpx
from ._base import LLMService

class LLAMA2Service(LLMService):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def completion(self, prompt: str, model = "llama2", **kwargs) -> AsyncIterator[str]:
        url = f"{self.base_url}/api/generate"
        body = {
            "stream": True, 
            "model": model, 
            "prompt": prompt,
            **kwargs,
        }
        
        with httpx.stream("POST", url, json=body, timeout=60) as res:
            for line in res.iter_lines():
                data = json.loads(line)
                if data.get("done", True):
                    return
                yield data.get("response")

    async def chat(self, messages: list[dict], model="llama2", **kwargs) -> AsyncIterator[str]:
        url = f"{self.base_url}/api/chat"
        body = {
            "messages": messages,
            "stream": True,
            "model": model,
            **kwargs,
        }
        
        with httpx.stream("POST", url, json=body, timeout=60) as res:
            for line in res.iter_lines():
                data = json.loads(line)
                if data.get("done", True):
                    return
                yield data.get("message", {}).get("content")

    async def calculate_embedding(self, prompt: str) -> list[float]:
        url = f'{self.base_url}/api/embeddings'
        body = {"model": "llama2", "prompt": prompt}
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(url, json=body, timeout=60)
                return res.json().get("embedding") or []
        except TimeoutError:
            return []
