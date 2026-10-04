# Serving image: the pricing engine + trader API. Model fitting (PyMC) is offline and not
# shipped; the engine reads the reviewed config/model.json and a pregame slate file.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[kafka]"

COPY config ./config
COPY artifacts/pregame_slate.json ./artifacts/pregame_slate.json

RUN useradd --create-home app
USER app

ENV LIVEPRICING_SERVE=1
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "livepricing.service:app", "--host", "0.0.0.0", "--port", "8000"]
