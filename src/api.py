import asyncio
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from scipy import spatial
import numpy as np
from ollama import Ollama

app = FastAPI()
llama2 = Ollama("http://localhost:11434")

class Request(BaseModel):
    query: str
    document: str
    k: int

@app.post("/")
async def main(body: Request):
    sentences = body.document.split('.')
    sentences = [item.strip() for item in sentences]

    [query_embedding, *embeddings] = await asyncio.gather(*[
        llama2.calculate_embeddings(item) for item in [body.query, *sentences]])
    
    filtered = filter(lambda x: len(x[0]) > 0, zip(embeddings, sentences))
    
    scores = list(
        map(lambda x: (
            x[1], 
            spatial.distance.cosine(
                np.array(x[0]).reshape(-1),
                np.array(query_embedding).reshape(-1)
            )
        ), filtered)
    )
    
    result = sorted(scores, key=lambda item: item[1])
    result.reverse()
    
    return JSONResponse({ "results": list(map(lambda x: {"text": x[0], "score": x[1]}, result[:body.k])) })

class ChatBody(BaseModel):
    sentences: list[[str, float]]
    query: str

@app.post("/chat")
async def chat(body: ChatBody):
    async def generator():
        async for text in llama2.chat(list(map(lambda x: {"content": x[0], "role": "user"}, body.sentences))):
            yield text
    return EventSourceResponse(generator())

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api:app", reload=True)