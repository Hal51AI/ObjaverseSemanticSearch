from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    CAPTIONS_FILE: str = "data/pali_captions_with_likes.csv"
    EMBEDDINGS_FILE: str = "data/embeddings.npy"
    DATABASE_PATH: str = "data/database.sqlite3"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    SIMILARITY_SEARCH: str = "IVFSimilarity"

    class Config:
        env_file = ".env"


settings = Settings()
