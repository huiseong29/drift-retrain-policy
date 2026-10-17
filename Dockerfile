FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY drift.py policy.py ./
COPY service/ ./service/

ENV PYTHONUNBUFFERED=1 \
    DRIFT_DB=/data/pipeline.db \
    DRIFT_MODELS=/data/models

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health').status_code==200 else 1)"

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]