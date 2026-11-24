# Multi-stage image.
#   serve (default): API + Streamlit runtime with the trained artifacts baked in.
#   train:           full environment that runs training + one-time evaluation.
#
#   docker build -t smart-health-diabetes .                       # serve image
#   docker run -p 8000:8000 smart-health-diabetes                 # FastAPI on :8000
#   docker run -p 8501:8501 smart-health-diabetes \
#       streamlit run app/streamlit_app.py --server.address 0.0.0.0   # UI on :8501
#   docker build --target train -t smart-health-diabetes:train .  # training image

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MLFLOW_DISABLE_AGENT_HINT=1
# libgomp1: OpenMP runtime required by XGBoost/CatBoost wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home appuser
WORKDIR /app

FROM base AS train
COPY requirements-serve.txt requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
CMD ["sh", "-c", "python scripts/download_data.py && python scripts/train.py && python scripts/evaluate.py"]

FROM base AS serve
COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt
COPY src ./src
COPY api ./api
COPY app ./app
COPY configs ./configs
COPY data ./data
COPY models ./models
COPY reports ./reports
RUN chown -R appuser /app
USER appuser
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
