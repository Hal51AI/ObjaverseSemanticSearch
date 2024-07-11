from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class SimilarityBase(ABC):
    """
    Abstract Base Class for a similarity search object to be used for API.
    """

    @classmethod
    def from_embeddings(
        cls,
        captions_file: str,
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
            embeddings=embeddings,
            sentence_transformer_model=sentence_transformer_model,
        )

    @abstractmethod
    async def search(self, query: str, top_k: int = 10) -> Dict[str, np.float32]:
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

    @abstractmethod
    async def download(self, query: str) -> Dict[str, str]:
        """
        Method for downloading the most similar glb file from the similarity score.
        Returns a dictionary where the keys are the glb file id's and the values
        are a path to a local file containing the downloaded glb file.

        Parameters
        ==========
        query: str
            The query string to search and download

        Returns
        =======
        Dict[str, str]
            A random file from the top matching query result
        """
