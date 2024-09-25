from __future__ import annotations

import logging
import os
import sqlite3
from typing import TYPE_CHECKING, Any, Dict, Sized

import numpy as np
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool

if TYPE_CHECKING:
    from .abc import SimilarityBase

logger = logging.getLogger("uvicorn")


def create_embeddings(
    database_path: str, embeddings_file: str, sentence_transformer_model: str
) -> None:
    """
    Create sentence embeddings from a CSV file of captions and save them to a file.

    Parameters
    ==========
    database_path: str
        Path to the database file to query for features use to generate embeddings
    embeddings_path: str
        A file path to save to embeddings to
    sentence_transformer_model: str
        The name of the sentence transformer model to build embeddings from.
    """
    with sqlite3.connect(database_path) as db:
        res = db.execute("""
            SELECT rowid, top_aggregate_caption || ', ' || name || ', ' || description
            FROM combined
            ORDER BY rowid
        """).fetchall()
        full_captions = [i[1] for i in res]

    model = SentenceTransformer(sentence_transformer_model)

    logger.info("Creating embeddings")
    embeddings = model.encode(full_captions, show_progress_bar=True)

    logger.info(f"Saving Embeddings to {embeddings_file}")
    np.save(embeddings_file, embeddings)


async def create_similarity_model(
    captions_file: str,
    database_path: str,
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
    database_path: str
        The path to the database file containing captions and objaverse metadata
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
        await run_in_threadpool(
            create_embeddings,
            database_path=database_path,
            embeddings_file=embeddings_file,
            sentence_transformer_model=sentence_transformer_model,
        )

    similarity_model_cls = getattr(similarity, similarity_search)

    sim_model = await run_in_threadpool(
        similarity_model_cls.from_embeddings,
        captions_file=captions_file,
        database_path=database_path,
        embeddings_file=embeddings_file,
        sentence_transformer_model=sentence_transformer_model,
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


def reformat_results(data: Dict) -> Dict[str, Any]:
    """
    Reformat a dictionary by keeping specific keys at the top level
    and moving the remaining keys into a nested 'metadata' dictionary.

    Parameters
    ==========
    data: Dict:
        The original dictionary to be reformatted.

    Returns
    =======
    Dict:
        A new dictionary with 'object_uid', 'top_aggregate_caption',
        'probability' and 'similarity' as top-level keys, and the remaining keys
        inside a nested 'metadata' dictionary.

    Raises
    ======
    ValueError
        If the keys do not exist in the original dictionary.
    """
    top_level_keys = {
        "object_uid",
        "top_aggregate_caption",
        "probability",
        "similarity",
        "download_url",
    }
    new_dict = {key: data[key] for key in top_level_keys}
    new_dict["metadata"] = {
        key: value for key, value in data.items() if key not in top_level_keys
    }
    return new_dict
