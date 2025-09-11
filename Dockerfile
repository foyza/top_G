FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
