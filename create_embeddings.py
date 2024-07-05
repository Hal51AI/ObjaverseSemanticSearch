import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.config import settings


def create_embeddings() -> None:
    if os.path.isfile(settings.EMBEDDINGS_FILE):
        print(f"File already exists at {settings.EMBEDDINGS_FILE}")
        return

    df = pd.read_csv(settings.CAPTIONS_FILE, delimiter=";")
    model = SentenceTransformer("bert-base-nli-mean-tokens")

    print("Creating embeddings")
    embeddings = model.encode(list(df.top_aggregate_caption))

    print(f"Saving Embeddings to {settings.EMBEDDINGS_FILE}")
    np.save(settings.EMBEDDINGS_FILE, embeddings)


if __name__ == "__main__":
    create_embeddings()
