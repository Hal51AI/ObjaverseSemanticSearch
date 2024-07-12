from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Sized

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from .abc import SimilarityBase
logger = logging.getLogger("uvicorn")


def create_embeddings(
    captions_file: str, embeddings_file: str, sentence_transformer_model: str
) -> None:
    """
    Create sentence embeddings from a CSV file of captions and save them to a file.

    Parameters
    ==========
    captions_file: str
        The file path to the captions data. Must a semicolon delimeted.
    embeddings_path: str
        A file path to save to embeddings to
    sentence_transformer_model: str
        The name of the sentence transformer model to build embeddings from.
    """
    df = pd.read_csv(captions_file, delimiter=";")
    model = SentenceTransformer(sentence_transformer_model)

    logger.info("Creating embeddings")
    embeddings = model.encode(list(df.top_aggregate_caption), show_progress_bar=True)

    logger.info(f"Saving Embeddings to {embeddings_file}")
    np.save(embeddings_file, embeddings)


def create_similarity_model(
    captions_file: str,
    embeddings_file: str,
    sentence_transformer_model: str,
    similarity_search: str,
) -> SimilarityBase:
    """
    Create or load a SimilarityBase model based on the provided embeddings and captions.

    This function either creates a new SimilarityBase object and saves its embeddings,
    or loads an existing model from saved embeddings. It ensures that the necessary
    files exist before processing.

    Parameters
    ==========
    captions_file: str
        The file path to the captions data used for creating the similarity model.
    embeddings_path: str
        The file path where embeddings should be saved or loaded from.
    sentence_transformer_model: str
        A string for the sentence transformer model to load.
        Find all the different models that you can use here.
            https://sbert.net/docs/sentence_transformer/pretrained_models.html#original-models
    similarity_search: str
        The name of a concrete implementation of the SimilarityBase class to instantiate

    Returns
    =======
    SimilarityBase
        An concrete instance of the SimilarityBase model

    Raises
    ======
    FileNotFoundError
        If the specified captions file does not exist.
    """
    from app import similarity

    if not os.path.isfile(captions_file):
        raise FileNotFoundError(f"Could not find captions file at {captions_file}")

    if not os.path.isfile(embeddings_file):
        create_embeddings(captions_file, embeddings_file, sentence_transformer_model)

    similarity_model_cls = getattr(similarity, similarity_search)
    sim_model = similarity_model_cls.from_embeddings(
        captions_file, embeddings_file, sentence_transformer_model
    )

    return sim_model


def check_compatibility(
    dataset: Sized, embedding: np.ndarray, model: SentenceTransformer
) -> None:
    """
    Check the compatibility between a dataset, its embeddings, and
    the sentence transformer model output layer.

    This function ensures that the dataset size matches the size of the
    embeddings and that the embedding dimensions are compatible with the
    model's expected embedding dimensions.

    Parameters
    ----------
    dataset: Sized
        The dataset to be checked.
    embedding: np.ndarray
        The numpy array containing the embeddings.
    model: SentenceTransformer
        The sentence transformer model used to generate embeddings.

    Raises
    ------
    ValueError
        If the size of the dataset does not match the size of the embeddings,
        or if the embedding dimensions do not match the model's expected
        embedding dimensions.
    """
    if not len(dataset) == len(embedding):
        raise ValueError(
            " ".join(
                [
                    "Mismatch between embedding and dataset size.",
                    f"Dataset size: {len(dataset)}, Embedding size: {len(embedding)}.",
                    "Please rebuild embeddings by deleting embeddings npy file running again.",
                ]
            )
        )

    if not embedding.shape[-1] == model.get_sentence_embedding_dimension():
        raise ValueError(
            " ".join(
                [
                    "Mismatch between model embedding dimension and loaded embeddings.",
                    f"Model Dimension: {model.get_sentence_embedding_dimension()},",
                    f"Loaded Embedding Shape: {embedding.shape[-1]}",
                    "Please rebuild embeddings by deleting embeddings npy file running again.",
                ]
            )
        )
