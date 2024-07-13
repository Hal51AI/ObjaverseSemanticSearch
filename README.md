# Objaverse Semantic Search

[![mypy](https://github.com/Hal51AI/ObjaverseSemanticSearch/actions/workflows/mypy.yml/badge.svg)](https://github.com/Hal51AI/ObjaverseSemanticSearch/actions/workflows/mypy.yml)

## Install Dependencies

You can install dependencies using pip

```bash
pip install -r requirements.txt
```

## Create Embeddings

First you must create an embeddings file. By default it will be saved to `./data/embeddings.npy`, but you can change the location by modifying the `EMBEDDINGS_FILE` variable in the `.env` file.

```bash
python create_embeddings.npy
```

There are 700k descriptions for embeddings generation, so please be patient, it can take up to an hour depending on the amount of compute power you have.

If embeddings aren't created, the app will automatically generate the file before running. But the most effective way is to create the embeddings beforehand. 

## Run Locally

Once all dependencies are installed and embeddings are generated, the easiest way to run a local environment is to run

```bash
fastapi dev
```

## Run on Docker

To build with docker, we can run

```bash
docker build -t objaverse-semantic-search .
```

Then to run it, you can run

```bash
docker run -it -v ./data:/app/data -p 8000:8000 objaverse-semantic-search
```

All these operations assume you will have a pre-built embeddings file and it is stored at `./data`

## Run on Docker Compose

The easiest way to run is with docker compose, run

```bash
docker compose up
```