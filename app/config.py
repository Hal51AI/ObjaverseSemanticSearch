from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    CAPTIONS_FILE: str = "data/pali_captions.csv"
    EMBEDDINGS_FILE: str = "data/embeddings.npy"

    class Config:
        env_file = ".env"


settings = Settings()
