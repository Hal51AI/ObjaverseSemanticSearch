# Objaverse Semantic Search

[![mypy](https://github.com/Hal51AI/ObjaverseSemanticSearch/actions/workflows/mypy.yml/badge.svg)](https://github.com/Hal51AI/ObjaverseSemanticSearch/actions/workflows/mypy.yml)

## Install Dependencies

You can install dependencies using pip

```bash
pip install -r requirements.txt
```

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