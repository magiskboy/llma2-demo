from typing import Type
import asyncio
from dataclasses import dataclass
from ._base import Computation
from ..distances import DistanceCalculator
from ..embeddings import Embedding
from ..llm import LLAMA2Service
from ..utils import split_sentences

@dataclass()
class ScoredSentences:
    score: float
    text: str

class MatchingModel(Computation):
    def __init__(self, embedding: Embedding, distance: DistanceCalculator):
        self.embedding = embedding
        self.distance = distance

    async def compute(self, query: str, text: str) -> ScoredSentences:
        sentences = split_sentences(text)
        
        [query_vector, *sentence_vectors] = await asyncio.gather(
            *[self.embedding.compute(sentence) for sentence in ([query] + sentences)])

        scored: list[ScoredSentences] = []
        for vector, sentence in zip(sentence_vectors, sentences):
            score = self.distance.compute(query_vector, vector)
            scored.append(ScoredSentences(score, sentence))

        scored.sort(key=lambda item: -item.score)
        return scored
