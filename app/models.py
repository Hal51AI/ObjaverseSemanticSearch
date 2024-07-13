from pydantic import BaseModel


class ObjaverseDownloadItem(BaseModel):
    uid: str
    data: str
