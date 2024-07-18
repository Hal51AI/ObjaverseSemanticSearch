import base64
import os
import random
from contextlib import asynccontextmanager
from typing import Annotated, List

import aiofiles
import objaverse
from fastapi import FastAPI, HTTPException, Query, Response
from scipy.special import softmax
from starlette.concurrency import run_in_threadpool

from .config import settings
from .db import query_db_match
from .models import ObjaverseDownloadItem, ObjaverseSimilarityResult
from .utils import create_similarity_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Create a model and bind it to the fastapi object.
    When app is shutdown, also deletes the created database object
    as a cleanup action.

    Parameters
    ==========
    app: FastAPI
        Application object from fastapi
    """
    app.state.model = create_similarity_model(
        settings.CAPTIONS_FILE,
        settings.EMBEDDINGS_FILE,
        settings.SENTENCE_TRANSFORMER_MODEL,
        settings.SIMILARITY_SEARCH,
    )
    yield
    if os.path.exists(app.state.model.db_path):
        os.unlink(app.state.model.db_path)


description = """
## Description

The Objaverse Semantic Search API is a tool designed to enhance the discovery and utilization of assets within Objaverse.
This API uses a vector database and similarity search algorithms to find relevant 3D assets

## Endpoints

| Path          | Description                                                                   |
|---------------|-------------------------------------------------------------------------------|
| `/download`   | Directly download one or many items from objaverse if you know the object uid |
| `/similarity` | Perform similarity search over a query and grab relevant metadata             |
| `/glb`        | Download a single glb from your query                                         |

"""

app = FastAPI(
    title="Objaverse Semantic Search API",
    summary="Perform semantic search over objaverse and download 3d models",
    description=description,
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
        app.state.model.db_path, match_list=objaverse_ids, col_name="object_uid"
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
                            "match": "a polar bear",
                            "similarity": 0.7350118160247803,
                            "items": [
                                {
                                    "object_uid": "b41b41a2858046fea0021f677dc010c4",
                                    "top_aggregate_caption": "a polar bear",
                                    "probability": 0.4412872404285839,
                                },
                                {
                                    "object_uid": "cdd861d7849440abb95fa8e37376d099",
                                    "top_aggregate_caption": "a polar bear",
                                    "probability": 0.2862682242309294,
                                },
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
    match_df = await query_db_match(app.state.model.db_path, list(results))

    # Pack records into a datastructure to return to user
    records = []
    for match, group_df in sorted(
        match_df.groupby("top_aggregate_caption"),
        key=lambda x: results[x[0]],
        reverse=True,
    ):
        similarity = results[match]
        items = group_df.to_dict(orient="records")
        records.append({"match": match, "similarity": similarity, "items": items})

    return records


@app.get(
    "/glb",
    response_class=Response,
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
    match_df = await query_db_match(app.state.model.db_path, list(results))

    # Grab a random item from the objects weighted by the softmax probability
    weights = softmax(
        match_df.top_aggregate_caption.map(results) * match_df.probability
    )
    selection = random.choices(match_df.object_uid.tolist(), weights=weights)

    # Download from objaverse
    glb_map = await run_in_threadpool(objaverse.load_objects, selection)

    # read file from filesystem
    filepath = list(glb_map.values())[0]
    async with aiofiles.open(filepath, mode="rb") as fp:
        file_bytes = await fp.read()

    return Response(file_bytes, media_type="model/gltf-binary")
