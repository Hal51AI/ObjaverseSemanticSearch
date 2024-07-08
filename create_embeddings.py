import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.config import settings


def create_embeddings() -> None:
    if os.path.isfile(settings.EMBEDDINGS_FILE):
        print(f"File already exists at {settings.EMBEDDINGS_FILE}", file=sys.stderr)
        return None

    if not os.path.isfile(settings.CAPTIONS_FILE):
        print(
            f"Could not find captions file at {settings.CAPTIONS_FILE}", file=sys.stderr
        )
        return None

    df = pd.read_csv(settings.CAPTIONS_FILE, delimiter=";")
    model = SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)

    print("Creating embeddings")
    embeddings = model.encode(list(df.top_aggregate_caption), show_progress_bar=True)

    print(f"Saving Embeddings to {settings.EMBEDDINGS_FILE}")
    np.save(settings.EMBEDDINGS_FILE, embeddings)


if __name__ == "__main__":
    create_embeddings()
