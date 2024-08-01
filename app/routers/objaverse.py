import base64
from typing import Annotated, List

import aiofiles
import objaverse
from fastapi import APIRouter, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from ..db import query_db_match
from ..models import ObjaverseDownloadItem

router = APIRouter()


@router.get(
    "/download",
    response_model=List[ObjaverseDownloadItem],
    response_description="A list of base64 encoded glb files",
    responses={
        200: {
            "description": "GLB files requested by UID's",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "uid": "81b98c35166d4f75b559438a93843a71",
                            "data": "Z2xURgIAAAAgrA0AyBAAAEpTT057ImFjY2Vzc29ycyI6W3siYnVmZmVyVmlldyI6MiwiY29tcG9...",
                        },
                        {
                            "uid": "d16afc21f6d6486da1f7b274ddf52129",
                            "data": "Z2xURgIAAAA4RQIAzAkAAEpTT057ImFjY2Vzc29ycyI6W3siYnVmZmVyVmlldyI6MSwiY29tcG9...",
                        },
                    ]
                }
            },
        },
        404: {
            "description": "One of the UID's provided did not exist in the objaverse repository",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "The following objaverse_ids do not exist: ['nonce']"
                    }
                }
            },
        },
    },
)
async def download(
    request: Request,
    objaverse_ids: Annotated[
        List[str],
        Query(
            ...,
            description="List of objaverse ids. You can use `/similarity` to find ids based on a query",
        ),
    ],
) -> List[ObjaverseDownloadItem]:
    """
    Directly download one or many items from objaverse if you know the object uid
    """
    # Perform validation to ensure items exist
    match_df = await query_db_match(
        request.app.state.model.database_path,
        match_list=objaverse_ids,
        col_name="object_uid",
    )
    object_uid_set = set(match_df["object_uid"])
    missing_items = [item for item in objaverse_ids if item not in object_uid_set]

    # Raise exception if any items are not present
    if any(missing_items):
        raise HTTPException(
            status_code=404,
            detail=f"The following objaverse_ids do not exist: {list(missing_items)}",
        )

    # Download from objaverse
    file_map = await run_in_threadpool(
        objaverse.load_objects, objaverse_ids, download_processes=len(objaverse_ids)
    )

    # base64 encode files
    encoded_files = []
    for uid, filepath in file_map.items():
        async with aiofiles.open(filepath, mode="rb") as fp:
            file_bytes = await fp.read()
            encoded_files.append(
                ObjaverseDownloadItem(uid=uid, data=base64.b64encode(file_bytes))
            )

    return encoded_files
