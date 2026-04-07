FROM python:3.11-slim

WORKDIR /app

# System deps for unstructured, lxml, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libmagic1 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY . .

EXPOSE 8000 8001

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
