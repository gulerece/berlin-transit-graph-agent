FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TRANSIT_GRAPH_PATH=/app/data/graph.json.gz

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[ollama,openai,anthropic]"
COPY data/graph.json.gz ./data/graph.json.gz

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000 8001
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "transit_graph.api:app", "--host", "0.0.0.0", "--port", "8000"]
