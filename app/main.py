import os
import pathlib
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from .config import settings
from .utils import create_similarity_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Create a model and bind it to the fastapi object.
    When app is shutdown, also deletes the created database object
    as a cleanup action.
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
def similarity(query: str, top_k: int = 10):
    results = app.state.model.search(query, top_k=top_k)

    pat = "|".join(list(results.keys()))
    matches = app.state.model.df.top_aggregate_caption.str.fullmatch(pat)
    match_df = app.state.model.df[matches]

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
def glb(query: str):
    result = app.state.model.download(query)
    filepath = pathlib.Path(list(result.values())[0])
    file_bytes = filepath.read_bytes()

    # remove downloaded file after reading
    shutil.rmtree(filepath.parent)

    return Response(file_bytes, media_type="model/gltf-binary")
