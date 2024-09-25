from typing import Optional

from pydantic import BaseModel


class ObjaverseDownloadItem(BaseModel):
    uid: str
    data: bytes


class ObjaverseMetadataResult(BaseModel):
    name: str
    staffpickedAt: Optional[str]
    viewCount: int
    likeCount: int
    animationCount: int
    description: str
    faceCount: int
    vertexCount: int
    license: str
    publishedAt: str
    createdAt: str
    isAgeRestricted: bool
    userId: str
    userName: str
    userProfile: str


class ObjaverseItemResult(BaseModel):
    object_uid: str
    top_aggregate_caption: str
    probability: float
    similarity: float
    download_url: str
    metadata: ObjaverseMetadataResult


class LicenseInfo(BaseModel):
    label: str
    fullName: str
    requirements: str
    url: str
    slug: str
