import os
from contextlib import asynccontextmanager

import aiofiles
from fastapi import FastAPI, Response

from .config import settings
from .db import query_db_match
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
    )
    yield
    if os.path.exists(app.state.model.db_path):
        os.unlink(app.state.model.db_path)


app = FastAPI(
    title="ObjaverseSemanticSearch",
    summary="Perform semantic search over objaverse and download 3d models",
    lifespan=lifespan,
)


@app.get("/similarity")
async def similarity(query: str, top_k: int = 10):
    results = await app.state.model.search(query, top_k=top_k)
    match_df = await query_db_match(app.state.model.db_path, list(results))

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
    result = await app.state.model.download(query)
    filepath = list(result.values())[0]

    async with aiofiles.open(filepath, mode="rb") as fp:
        file_bytes = await fp.read()

    return Response(file_bytes, media_type="model/gltf-binary")
