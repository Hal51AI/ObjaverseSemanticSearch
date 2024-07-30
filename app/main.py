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
)
from .utils import create_similarity_model, reformat_results


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

    await create_db(settings.CAPTIONS_FILE, settings.DATABASE_PATH)

    app.state.model = await create_similarity_model(
        settings.CAPTIONS_FILE,
        settings.DATABASE_PATH,
        settings.EMBEDDINGS_FILE,
        settings.SENTENCE_TRANSFORMER_MODEL,
        settings.SIMILARITY_SEARCH,
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
    response_model=List[ObjaverseItemResult],
    responses={
        200: {
            "description": "A similarity query",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "object_uid": "2f0598cba938424688cdd048a90b8339",
                            "top_aggregate_caption": "a boat",
                            "probability": 0.439652563952344,
                            "similarity": 0.7142868041992188,
                            "metadata": {
                                "name": "boat",
                                "staffpickedAt": None,
                                "viewCount": 1775,
                                "likeCount": 28,
                                "animationCount": 0,
                                "description": "",
                                "faceCount": 18458,
                                "vertexCount": 10306,
                                "license": "by",
                                "publishedAt": "2021-04-29 23:22:23.021191",
                                "createdAt": "2021-04-29 22:45:13.799783",
                                "isAgeRestricted": False,
                                "userId": "49dd921047ae4da28aeaaa213e9a5d8a",
                                "userName": "jondameron5",
                            },
                        },
                        {
                            "object_uid": "56148c53e9664ee683b598fadb457992",
                            "top_aggregate_caption": "a black sailboat",
                            "probability": 0.3841391686492714,
                            "similarity": 0.6926552653312683,
                            "metadata": {
                                "name": "Schooner",
                                "staffpickedAt": None,
                                "viewCount": 844,
                                "likeCount": 26,
                                "animationCount": 0,
                                "description": "",
                                "faceCount": 7995,
                                "vertexCount": 5317,
                                "license": "by",
                                "publishedAt": "2019-06-26 06:51:11.054891",
                                "createdAt": "2019-06-26 06:46:39.725488",
                                "isAgeRestricted": False,
                                "userId": "0c477a42672142baadcefe9cc41845d2",
                                "userName": "Tobias.De.Maine",
                            },
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

    Schema
    ======
    | Schema                  | Type   | Description                                          |
    |-------------------------|--------|------------------------------------------------------|
    | `object_uid`            | UUID   | Unique identifier for objaverse assets               |
    | `top_aggregate_caption` | String | Caption generated from classifier                    |
    | `probability`           | Float  | The confidence/probability score from the classifier |
    | `similarity`            | Float  | The similarity score between the query               |
    | `metadata`              | Dict   | Asset information metadata                           |

    Metadata
    ========
    | Schema            | Type             | Description                                                                         |
    |-------------------|------------------|-------------------------------------------------------------------------------------|
    | `name`            | String           | The original name of the asset made by the author                                   |
    | `staffpickedAt`   | Datetime or null | High quality assets which have been selected by staff                               |
    | `viewCount`       | Integer          | The amount of times asset was viewed by another user at the time of data collection |
    | `likeCount`       | Integer          | The amount of times asset was liked by another user at the time of data collection  |
    | `animationCount`  | Integer          | The number of animations for the asset                                              |
    | `description`     | String           | A longer description made by the author                                             |
    | `faceCount`       | Integer          | The number of faces of the asset                                                    |
    | `vertexCount`     | Integer          | The number of verticies in the asset                                                |
    | `license`         | String           | The type of license used by the asset                                               |
    | `publishedAt`     | Datetime         | The time when the asset was published to the world                                  |
    | `createdAt`       | Datetime         | The time for which the asset was first created                                      |
    | `isAgeRestricted` | Boolean          | Whether the asset has been flagged to contain age restricted content                |
    | `userId`          | UUID             | The sketchfab user id used to query for more user information                       |
    | `userName`        | String           | The skechfab user name                                                              |
    """
    results = await app.state.model.search(query, top_k=top_k)
    return list(map(reformat_results, results))


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

    # Grab a random item from the objects weighted by the similarity score
    weights = softmax([i["similarity"] for i in results])
    selection = random.choices([i["object_uid"] for i in results], weights=weights)

    # Download from objaverse
    glb_map = await run_in_threadpool(objaverse.load_objects, selection)

    # read file from filesystem
    filepath = list(glb_map.values())[0]

    return FileResponse(
        path=filepath,
        media_type="model/gltf-binary",
        filename=os.path.basename(filepath),
    )
