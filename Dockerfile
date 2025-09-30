# Базовый образ Python
FROM python:3.11-slim

# Установим системные зависимости (для numpy, pandas, sklearn, tensorflow, nltk и т.д.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libxml2 \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установим pip и обновим его
RUN pip install --upgrade pip setuptools wheel

# Скопируем requirements.txt
COPY requirements.txt .

# Установим Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Скачаем данные для nltk (например, токенайзер и стоп-слова)
RUN python -m nltk.downloader punkt stopwords

# Копируем весь проект в контейнер
WORKDIR /app
COPY . .

# Укажем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Запуск бота
CMD ["python", "main.py"]
