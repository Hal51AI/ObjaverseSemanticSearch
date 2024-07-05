FROM python:3.11-slim

ENV LANG C.UTF-8
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN : \
    && pip install --no-cache -r requirements.txt

COPY . /app

ENTRYPOINT ["gunicorn", "--bind", ":8000", "app.main:app", "--worker-class", "uvicorn.workers.UvicornH11Worker"]
