from contextlib import asynccontextmanager
from textwrap import dedent

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import create_db
from .routers import licenses, objaverse, similarity, users
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
    """),
    contact={"name": "Hal51 AI", "url": "https://github.com/hal51ai"},
    license_info={"name": "MIT LIcense", "identifier": "MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(similarity.router, prefix="/similarity", tags=["similarity"])
app.include_router(objaverse.router, prefix="/objaverse", tags=["objaverse"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(licenses.router, prefix="/licenses", tags=["licenses"])
