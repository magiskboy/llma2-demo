from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_host: str = Field(default="http://localhost:11434")

settings = Settings()
