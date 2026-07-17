FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv

COPY src ./src
COPY README.md ./README.md

ENV PYTHONPATH=/app/src
ENV NNM_CONVERTED_DIR=/app
ENV NNM_BACKEND_ARTIFACT_ROOT=/app/jobs

RUN uv sync --no-dev

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
