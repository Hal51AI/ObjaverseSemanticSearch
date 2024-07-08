import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .similarity import BERTSimilarityNN


def create_embeddings(
    captions_file: str, embeddings_file: str, sentence_transformer_model: str
) -> None:
    df = pd.read_csv(captions_file, delimiter=";")
    model = SentenceTransformer(sentence_transformer_model)

    print("Creating embeddings")
    embeddings = model.encode(list(df.top_aggregate_caption), show_progress_bar=True)

    print(f"Saving Embeddings to {embeddings_file}")
    np.save(embeddings_file, embeddings)


def create_similarity_model(
    captions_file: str, embeddings_file: str, sentence_transformer_model: str
) -> BERTSimilarityNN:
    """
    Create or load a BERTSimilarity model based on the provided embeddings and captions.

    This function either creates a new BERTSimilarity model and saves its embeddings,
    or loads an existing model from saved embeddings. It ensures that the necessary
    files exist before processing.

    Parameters
    ==========
    captions_file: str
        The file path to the captions data used for creating the similarity model.
    embeddings_path: str
        The file path where embeddings should be saved or loaded from.

    Returns
    =======
    BERTSimilarity
        An instance of the BERTSimilarity model, either newly created or loaded from existing embeddings.

    Raises
    ======
    FileNotFoundError
        If the specified captions file does not exist.
    """
    if not os.path.isfile(captions_file):
        raise FileNotFoundError(f"Could not find captions file at {captions_file}")

    if not os.path.isfile(embeddings_file):
        create_embeddings(captions_file, embeddings_file, sentence_transformer_model)

    sim_model = BERTSimilarityNN.from_embeddings(
        captions_file, embeddings_file, sentence_transformer_model
    )

    return sim_model
