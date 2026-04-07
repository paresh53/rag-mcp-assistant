FROM python:3.11-slim

WORKDIR /app

# System deps for unstructured, lxml, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libmagic1 poppler-utils tesseract-ocr \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxfixes3 \
    libnss3 libnspr4 libpango-1.0-0 libcairo2 libasound2t64 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e . && \
    pip install playwright && \
    python -m playwright install chromium

COPY . .

EXPOSE 9000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "${PORT:-9000}"]
