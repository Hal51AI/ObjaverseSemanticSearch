import asyncio
import base64
import os
import random
from contextlib import asynccontextmanager
from textwrap import dedent
from typing import Annotated, List

import aiofiles
import objaverse
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from scipy.special import softmax
from starlette.concurrency import run_in_threadpool

from .config import settings
from .db import create_db, query_db_match
from .models import (
    ObjaverseDownloadItem,
    ObjaverseItemResult,
    ObjaverseMetadataResult,
    ObjaverseSimilarityResult,
)
from .utils import create_similarity_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Create a model and bind it to the fastapi object while concurrently
    creating the database object.

    Parameters
    ==========
    app: FastAPI
        Application object from fastapi
    """

    app.state.model, _ = await asyncio.gather(
        create_similarity_model(
            settings.CAPTIONS_FILE,
            settings.DATABASE_PATH,
            settings.EMBEDDINGS_FILE,
            settings.SENTENCE_TRANSFORMER_MODEL,
            settings.SIMILARITY_SEARCH,
        ),
        create_db(settings.CAPTIONS_FILE, settings.DATABASE_PATH),
    )
    yield


app = FastAPI(
    title="Objaverse Semantic Search API",
    summary="Perform semantic search over objaverse and download 3d models",
    description=dedent("""
        ## Description

        The Objaverse Semantic Search API is a tool designed to enhance the discovery and utilization of assets within Objaverse.
        This API uses a vector database and similarity search algorithms to find relevant 3D assets

        ## Endpoints

        | Path          | Description                                                                   |
        |---------------|-------------------------------------------------------------------------------|
        | `/download`   | Directly download one or many items from objaverse if you know the object uid |
        | `/similarity` | Perform similarity search over a query and grab relevant metadata             |
        | `/glb`        | Download a single glb from your query                                         |
    """),
    contact={"name": "Hal51 AI", "url": "https://github.com/hal51ai"},
    license_info={"name": "MIT LIcense", "identifier": "MIT"},
    lifespan=lifespan,
)


@app.get(
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
    tags=["objaverse"],
)
async def download(
    objaverse_ids: Annotated[
        List[str],
        Query(
            ...,
            description="List of objaverse ids. You can use `/similarity` to find ids based on a query",
        ),
    ],
):
    """
    Directly download one or many items from objaverse if you know the object uid
    """
    # Perform validation to ensure items exist
    match_df = await query_db_match(
        app.state.model.database_path, match_list=objaverse_ids, col_name="object_uid"
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
            encoded_files.append({"uid": uid, "data": base64.b64encode(file_bytes)})

    return encoded_files


@app.get(
    "/similarity",
    tags=["query"],
    response_model=List[ObjaverseSimilarityResult],
    responses={
        200: {
            "description": "A similarity query",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "match": "a black bear",
                            "similarity": 0.8646828532218933,
                            "items": [
                                {
                                    "object_uid": "da9b588f7a7346519f391c3eb9532226",
                                    "top_aggregate_caption": "a black bear",
                                    "probability": 0.3426025195650379,
                                    "metadata": {
                                        "name": "Bear",
                                        "staffpickedAt": None,
                                        "viewCount": 855,
                                        "likeCount": 14,
                                        "animationCount": 0,
                                        "description": "Scientific name: Ursidae\nSpeed: Polar bear: 40 km",
                                        "faceCount": 12126,
                                        "vertexCount": 6083,
                                        "license": "by",
                                        "publishedAt": "2021-09-28 09:35:12.478873",
                                        "createdAt": "2021-09-28 09:31:59.137726",
                                        "isAgeRestricted": False,
                                        "userId": "9b1a1d4bacff44d28116cfe61bc5d164",
                                        "userName": "sdpm",
                                    },
                                }
                            ],
                        },
                        {
                            "match": "a cartoon bear",
                            "similarity": 0.8628631234169006,
                            "items": [
                                {
                                    "object_uid": "33e532df00c54a13beda1cea02cef604",
                                    "top_aggregate_caption": "a cartoon bear",
                                    "probability": 0.159121752762681,
                                    "metadata": {
                                        "name": "Another Grizzly Bear Turned Into Stone",
                                        "staffpickedAt": None,
                                        "viewCount": 430,
                                        "likeCount": 6,
                                        "animationCount": 0,
                                        "description": "There is another bear turned into stone.",
                                        "faceCount": 18530,
                                        "vertexCount": 10890,
                                        "license": "by",
                                        "publishedAt": "2018-06-30 12:46:25.321332",
                                        "createdAt": "2018-06-30 12:44:40.455319",
                                        "isAgeRestricted": False,
                                        "userId": "2d2b33f4ca0149cab59981982f39f30b",
                                        "userName": "marvelvsdcvscapcomvssega",
                                    },
                                }
                            ],
                        },
                    ]
                }
            },
        }
    },
)
async def similarity(
    query: Annotated[
        str, Query(description="Perform similarity search on the query string")
    ],
    top_k: Annotated[
        int, Query(description="Grab top k results based on similarity")
    ] = 10,
):
    """
    Perform similarity search over a query and grab relevant metadata
    """
    results = await app.state.model.search(query, top_k=top_k)
    match_df = await query_db_match(
        app.state.model.database_path, list(results), table_name="combined"
    )

    # Pack records into a datastructure to return to user
    records = []
    for match, group_df in sorted(
        match_df.groupby("top_aggregate_caption"),
        key=lambda x: results[x[0]],
        reverse=True,
    ):
        similarity = results[match]
        group_dict = group_df.to_dict(orient="records")
        items = [
            ObjaverseItemResult(metadata=dict(ObjaverseMetadataResult(**i)), **i)
            for i in group_dict
        ]

        records.append({"match": match, "similarity": similarity, "items": items})

    return records


@app.get(
    "/glb",
    response_class=FileResponse,
    responses={200: {"content": {"model/gltf-binary": {"example": "binary blob..."}}}},
    tags=["query"],
)
async def glb(
    query: Annotated[
        str, Query(description="Perform similarity search on the query string")
    ],
):
    """
    Perform similarity search over a query and grab relevant metadata
    """
    results = await app.state.model.search(query, top_k=100)
    match_df = await query_db_match(app.state.model.database_path, list(results))

    # Grab a random item from the objects weighted by the softmax probability
    weights = softmax(
        match_df.top_aggregate_caption.map(results) * match_df.probability
    )
    selection = random.choices(match_df.object_uid.tolist(), weights=weights)

    # Download from objaverse
    glb_map = await run_in_threadpool(objaverse.load_objects, selection)

    # read file from filesystem
    filepath = list(glb_map.values())[0]

    return FileResponse(
        path=filepath,
        media_type="model/gltf-binary",
        filename=os.path.basename(filepath),
    )
