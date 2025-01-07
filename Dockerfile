FROM python:3.11-slim

ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN --mount=type=bind,source=requirements.txt,target=/app/requirements.txt : \
    && pip install --no-cache -r requirements.txt

COPY . /app
COPY data/embeddings.npy /app/data/embeddings.npy
COPY data/database.sqlite3 /app/data/database.sqlite3

EXPOSE 8000

ENTRYPOINT ["gunicorn", "--bind", ":8000", "app.main:app", "--worker-class", "uvicorn.workers.UvicornH11Worker"]
