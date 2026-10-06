FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /lab
RUN apt-get update && apt-get install -y --no-install-recommends git curl build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
CMD ["bash", "-c", "python -m src.run_all --fast && python -m http.server 8000 --directory benchmarks/results"]
