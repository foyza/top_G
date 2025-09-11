# Используем Python 3.11 slim
FROM python:3.11-slim

# Устанавливаем зависимости системы
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем requirements
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# ENV переменные (можно позже заменить .env)
# ENV TELEGRAM_TOKEN=your_token
# ENV TWELVEDATA_API_KEY=your_key
# ENV NEWSAPI_KEY=your_news_key

# Запуск бота
CMD ["python", "main.py"]
