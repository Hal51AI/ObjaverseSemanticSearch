from typing import List

from pydantic import BaseModel


class ObjaverseDownloadItem(BaseModel):
    uid: str
    data: str


class ObjaverserObjectResult(BaseModel):
    object_uid: str
    top_aggregate_caption: str
    probability: float


class ObjaverseSimilarityResult(BaseModel):
    match: str
    similarity: float
    items: List[ObjaverserObjectResult]
