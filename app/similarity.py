import random
from typing import Dict, Optional

import faiss
import numpy as np
import objaverse
import pandas as pd
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool

from .abc import SimilarityBase
from .db import create_db, query_db_match
from .utils import check_compatibility


class BERTSimilarity(SimilarityBase):
    """
    This class is designed to find the most similar captions to a
    given query using a pre-trained BERT model. This class takes
    a file containing captions, computes embeddings for these captions,
    and allows for searching similar captions based on a query.

    The `captions_file` is a semicolon delimited file containing these fields.
        - object_uid: uid which cooresponds to an objaverse glb file
        - top_aggregate_caption: a caption description of the glb file
        - probability: the probability of the last hidden layer

    Parameters
    ==========
    captions_file: str
        The file path to the semicolon delimited file containing captions
    embeddings: np.ndarray, optional
        Precomputed embeddings for the captions.
        If not provided, embeddings will be computed using the BERT model.
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
        BERT model for generating embeddings.
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

    async def download(self, query: str) -> Dict[str, str]:
        """
        This method downloads a random selection from the most similar captions found
        through the similarity search. It then returns a dictionary containing
        the file location of the downloaded file.

        Parameters
        ==========
        query: str
            The query string to search and download

        Returns
        =======
        Dict[str, str]
            A random file from the top matching query result
        """
        results = await self.search(query, top_k=10)
        match_df = await query_db_match(self.db_path, list(results))
        weights = softmax(
            match_df.top_aggregate_caption.map(results) * match_df.probability
        )

        # grab a random item from the objects weighted by the softmax probability
        selection = random.choices(match_df.object_uid.tolist(), weights=weights)

        return await run_in_threadpool(objaverse.load_objects, selection)


class BERTSimilarityNN(SimilarityBase):
    """
    Uses faiss nearest neighbor search to determine similar captions
    rather than a exhaustive comparison between each element of
    the embeddings.

    Parameters
    ==========
    captions_file: str
        The file path to the semicolon delimited file containing captions
    embeddings: np.ndarray, optional
        Precomputed embeddings for the captions.
        If not provided, embeddings will be computed using the BERT model.
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
        BERT model for generating embeddings.
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

    async def download(self, query: str) -> Dict[str, str]:
        """
        This method downloads a random selection from the most similar captions found
        through the similarity search. It then returns a dictionary containing
        the file location of the downloaded file.

        Parameters
        ==========
        query: str
            The query string to search and download

        Returns
        =======
        Dict[str, str]
            A random file from the top matching query result
        """
        results = await self.search(query, top_k=100)
        match_df = await query_db_match(self.db_path, list(results))
        weights = softmax(
            match_df.top_aggregate_caption.map(results) * match_df.probability
        )

        # grab a random item from the objects weighted by the softmax probability
        selection = random.choices(match_df.object_uid.tolist(), weights=weights)

        return await run_in_threadpool(objaverse.load_objects, selection)
