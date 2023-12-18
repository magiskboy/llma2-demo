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
    def __init__(self, embedding: Type[Embedding], distance: Type[DistanceCalculator]):
        self.embedding_class = embedding
        self.distance_class = distance

    async def compute(self, query: str, text: str) -> ScoredSentences:
        sentences = split_sentences(text)
        
        embedding_strategy = self.embedding_class()
        [query_vector, *sentence_vectors] = await asyncio.gather(
            *[embedding_strategy.compute(sentence) for sentence in ([query] + sentences)])

        scored: list[ScoredSentences] = []
        distance_strategy = self.distance_class()
        for vector, sentence in zip(sentence_vectors, sentences):
            score = self.distance_class().compute(query_vector, vector)
            scored.append(ScoredSentences(score, sentence))

        scored.sort(key=lambda item: -item.score)
        return scored