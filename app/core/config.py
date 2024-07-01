from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Causal Inference API"
    PROJECT_VERSION: str = "0.1.0"
    PROJECT_DESCRIPTION: str = "An API for causal inference using various methods and tools."
    API_V1_STR: str = "/api/v1"

    class Config:
        env_file = ".env"

settings = Settings()