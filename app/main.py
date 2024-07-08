import pathlib
import shutil

from fastapi import FastAPI, Response

from .config import settings
from .utils import create_similarity_model

app = FastAPI(
    title="ObjaverseSemanticSearch",
    summary="Perform semantic search over objaverse and download 3d models",
)


sim_model = create_similarity_model(
    settings.CAPTIONS_FILE,
    settings.EMBEDDINGS_FILE,
    settings.SENTENCE_TRANSFORMER_MODEL,
)


@app.get("/similarity")
def similarity(query: str, top_k: int = 10):
    results = sim_model.search(query, top_k=top_k)

    pat = "|".join(list(results.keys()))
    matches = sim_model.df.top_aggregate_caption.str.fullmatch(pat)
    match_df = sim_model.df[matches]

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
    result = sim_model.download(query)
    filepath = pathlib.Path(list(result.values())[0])
    file_bytes = filepath.read_bytes()

    # remove downloaded file after reading
    shutil.rmtree(filepath.parent)

    return Response(file_bytes, media_type="model/gltf-binary")
