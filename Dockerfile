FROM python:3.11-slim

# Установим зависимости для сборки некоторых пакетов
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
