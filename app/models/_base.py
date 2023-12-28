from typing import Protocol

class Computation(Protocol):
    async def compute(self):
        ...
