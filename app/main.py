import base64
import os
import random
from contextlib import asynccontextmanager
from typing import List

import aiofiles
import objaverse
from fastapi import FastAPI, HTTPException, Query, Response
from scipy.special import softmax
from starlette.concurrency import run_in_threadpool

from .config import settings
from .db import query_db_match
from .similarity import IVFSimilarity
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
        IVFSimilarity,
    )
    yield
    if os.path.exists(app.state.model.db_path):
        os.unlink(app.state.model.db_path)


app = FastAPI(
    title="ObjaverseSemanticSearch",
    summary="Perform semantic search over objaverse and download 3d models",
    lifespan=lifespan,
)


@app.get("/download")
async def download(
    objaverse_ids: List[str] = Query(..., description="List of objaverse ids"),
):
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
    encoded_files = {}
    for uid, filepath in file_map.items():
        async with aiofiles.open(filepath, mode="rb") as fp:
            file_bytes = await fp.read()
            encoded_files[uid] = base64.b64encode(file_bytes)

    return encoded_files


@app.get("/similarity")
async def similarity(query: str, top_k: int = 10):
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
    responses={200: {"content": {"model/gltf-binary": {}}}},
)
async def glb(query: str):
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
