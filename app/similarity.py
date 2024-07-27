from typing import Any, Dict, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool

from .abc import SimilarityBase
from .db import query_db_match
from .utils import check_compatibility


class BruteForceSimilarity(SimilarityBase):
    """
    This class uses a brute force method and compares similarity between
    the query and evey element of the embedding to determine the nearest
    neighbors.

    Parameters
    ==========
    captions_file: str
        The file path to the semicolon delimited file containing captions
    database_path: str
        Location of the populated database file
    embeddings: np.ndarray
        Precomputed embeddings for the captions.
    sentence_transformer_model: str
        The name of the model to use from sentence transformers

    Attributes
    ==========
    captions_file: str
        The file path to the captions CSV file
    database_path: str
        Location of the populated database file
    df: pandas.DataFrame
        DataFrame containing the data from the CSV file.
    sentence_transformer_model: str
        The name of the model used from sentence transformers
    model: SentenceTransformer
        SentenceTransformer model for generating embeddings.
    embeddings: np.ndarray
        Embeddings for the captions.
    """

    def __init__(
        self,
        captions_file: str,
        database_path: str,
        embeddings: np.ndarray,
        sentence_transformer_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.captions_file = captions_file
        self.database_path = database_path
        self.embeddings = embeddings
        self.df = pd.read_csv(captions_file, delimiter=";")
        self.sentence_transformer_model = sentence_transformer_model
        self.model = SentenceTransformer(sentence_transformer_model)

        check_compatibility(self.df, self.embeddings, self.model)

    async def search(self, query: str, top_k: int = 10) -> Dict[str, float]:
        """
        Async version of search function. Runs the similarity search under
        a threadpool

        Parameters
        ==========
        query: str
            The query string to search for similar captions.
        top_k: int
            The amount of results to show

        Returns
        =======
        Dict[str, float]
            The top captions and their similarity scores
        """

        key_map = await run_in_threadpool(self.search_sync, query, top_k)

        db_indices = list(map(str, key_map.keys()))
        result_df = await query_db_match(
            self.database_path,
            match_list=db_indices,
            table_name="combined",
            col_name="rowid",
        )

        result_df["similarity"] = list(key_map.values())

        return result_df.drop("rowid", axis=1).to_dict(orient="records")

    def search_sync(self, query: str, top_k: int = 10) -> Dict[int, float]:
        """
        This method searches for the most similar captions to a given query
        and returns a dictionary containing the index and their similarity scores.

        Parameters
        ==========
        query: str
            The query string to search for similar captions.
        top_k: int
            The amount of results to show

        Returns
        =======
        Dict[int, float]
            The top captions and their similarity scores
        """
        qx = np.array(self.model.encode([query]))
        sim_arr = np.array(self.model.similarity(qx, self.embeddings))[0]

        di = dict(zip(sim_arr, self.df.index + 1))
        top_arr = np.sort(np.unique(sim_arr))[::-1][:top_k]
        return {di[i]: float(i) for i in top_arr}


class IVFSimilarity(SimilarityBase):
    """
    Uses faiss inverted file index to find similar captions.

    Parameters
    ==========
    captions_file: str
        The file path to the semicolon delimited file containing captions
    database_path: str
        Location of the populated database file
    embeddings: np.ndarray
        Precomputed embeddings for the captions.
    sentence_transformer_model: str
        The name of the model to use from sentence transformers

    Attributes
    ==========
    captions_file: str
        The file path to the captions CSV file
    database_path: str
        Location of the populated database file
    df: pandas.DataFrame
        DataFrame containing the data from the CSV file.
    sentence_transformer_model: str
        The name of the model used from sentence transformers
    model: SentenceTransformer
        SentenceTranformer model for generating embeddings.
    quantizer: faiss.IndexFlatIP
        Faiss quantizer using inner product
    index: faiss.IndexIVFFlat
        Faiss IVF index which computes nearest neighbor search
    """

    def __init__(
        self,
        captions_file: str,
        database_path: str,
        embeddings: np.ndarray,
        sentence_transformer_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.captions_file = captions_file
        self.database_path = database_path
        self.df = pd.read_csv(captions_file, delimiter=";")
        self.sentence_transformer_model = sentence_transformer_model
        self.model = SentenceTransformer(sentence_transformer_model)

        self.quantizer = faiss.IndexScalarQuantizer(
            embeddings.shape[-1],
            faiss.ScalarQuantizer.QT_fp16,
            faiss.METRIC_INNER_PRODUCT,
        )
        self.index = faiss.IndexIVFFlat(
            self.quantizer, embeddings.shape[-1], 512, faiss.METRIC_INNER_PRODUCT
        )
        faiss.normalize_L2(embeddings)
        self.index.train(embeddings)
        self.index.add(embeddings)
        self.index.nprobe = 16

        check_compatibility(self.df, embeddings, self.model)

    async def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Finds the most similar captions to a given query and returns the

        Parameters
        ==========
        query: str
            The query string to search for similar captions.
        top_k: int
            The amount of results to show

        Returns
        =======
        Dict[str, float]
            The top captions and their similarity scores
        """
        dist, ind = await run_in_threadpool(self.search_index, query, top_k)

        db_indices = list((ind[0] + 1).astype(str))
        result_df = await query_db_match(
            self.database_path, db_indices, table_name="combined", col_name="rowid"
        )
        result_df["similarity"] = dist[0]

        return result_df.drop("rowid", axis=1).to_dict(orient="records")

    def search_index(
        self, query: str, top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method searches for the most similar captions to a given query
        and returns tuple containing their distance scores and indices.

        Parameters
        ==========
        query: str
            The query string to search for similar captions.
        top_k: int
            The amount of results to show

        Returns
        =======
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the distance and indices
        """
        qx = self.model.encode([query])
        faiss.normalize_L2(qx)

        return self.index.search(qx, top_k)
