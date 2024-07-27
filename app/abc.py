from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class SimilarityBase(ABC):
    """
    Abstract Base Class for a similarity search object to be used for API.
    """

    @abstractmethod
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

    @classmethod
    def from_embeddings(
        cls,
        captions_file: str,
        database_path: str,
        embeddings_file: str,
        sentence_transformer_model: str = "all-MiniLM-L6-v2",
    ) -> "SimilarityBase":
        """
        Instantiate a class by loading embeddings from a file

        Parameters
        ==========
        captions_file: str
            The file path to the captions CSV file
        embeddings_file:
            The file containing embeddings as an npy file
        """
        embeddings = np.load(embeddings_file).astype(np.float32)
        return cls(
            captions_file,
            database_path=database_path,
            embeddings=embeddings,
            sentence_transformer_model=sentence_transformer_model,
        )

    @abstractmethod
    async def search(self, query: str, top_k: int = 10) -> Dict[str, float]:
        """
        Method for searching similar captions. It must return a dictionary where the
        keys are the matching queries as a string and the values are similarity
        scores which are probability values from 0 to 1.

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
