from typing import Dict, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool

from .abc import SimilarityBase
from .db import create_db
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
    embeddings: np.ndarray, optional
        Precomputed embeddings for the captions.
    sentence_transformer_model: str
        The name of the model to use from sentence transformers

    Attributes
    ==========
    captions_file: str
        The file path to the captions CSV file
    df: pandas.DataFrame
        DataFrame containing the data from the CSV file.
    db_path: str
        A path to the sqlite3 database file created from dataframe
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
        embeddings: Optional[np.ndarray] = None,
        sentence_transformer_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.captions_file = captions_file
        self.df = pd.read_csv(captions_file, delimiter=";")
        self.db_path = create_db(self.df)
        self.sentence_transformer_model = sentence_transformer_model
        self.model = SentenceTransformer(sentence_transformer_model)
        if isinstance(embeddings, np.ndarray):
            self.embeddings = embeddings
        else:
            self.embeddings = self.model.encode(
                list(self.df.top_aggregate_caption), show_progress_bar=True
            )

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
        return await run_in_threadpool(self.search_sync, query, top_k)

    def search_sync(self, query: str, top_k: int = 10) -> Dict[str, float]:
        """
        This method searches for the most similar captions to a given query
        and returns a dictionary containing the captions and their similarity scores.

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
        qx = self.model.encode([query])
        sim_arr = np.array(self.model.similarity(qx, self.embeddings))[0]

        di = dict(zip(sim_arr, self.df.top_aggregate_caption))
        top_arr = np.sort(np.unique(sim_arr))[::-1][:top_k]

        return {di[i]: float(i) for i in top_arr}


class IVFSimilarity(SimilarityBase):
    """
    Uses faiss inverted file index to find similar captions.

    Parameters
    ==========
    captions_file: str
        The file path to the semicolon delimited file containing captions
    embeddings: np.ndarray, optional
        Precomputed embeddings for the captions.
    sentence_transformer_model: str
        The name of the model to use from sentence transformers

    Attributes
    ==========
    captions_file: str
        The file path to the captions CSV file
    df: pandas.DataFrame
        DataFrame containing the data from the CSV file.
    db_path: str
        A path to the sqlite3 database file created from dataframe
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
        embeddings: Optional[np.ndarray] = None,
        sentence_transformer_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.captions_file = captions_file
        self.df = pd.read_csv(captions_file, delimiter=";")
        self.db_path = create_db(self.df)
        self.sentence_transformer_model = sentence_transformer_model
        self.model = SentenceTransformer(sentence_transformer_model)

        if not isinstance(embeddings, np.ndarray):
            embeddings = self.model.encode(
                list(self.df.top_aggregate_caption),
                show_progress_bar=True,
            )

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
        return await run_in_threadpool(self.search_sync, query, top_k)

    def search_sync(self, query: str, top_k: int = 10) -> Dict[str, float]:
        """
        This method searches for the most similar captions to a given query
        and returns a dictionary containing the captions and their similarity scores.

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
        qx = self.model.encode([query])
        faiss.normalize_L2(qx)

        dist, ind = self.index.search(qx, top_k * 10)
        found_captions = [self.df.top_aggregate_caption[i] for i in ind[0]]

        if len(set(found_captions)) < top_k:
            dist, ind = self.index.search(qx, top_k * 100)
            found_captions = [self.df.top_aggregate_caption[i] for i in ind[0]]

        results = dict(zip(found_captions, [float(i) for i in dist[0]]))
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k])
