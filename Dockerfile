FROM python:3.11-slim

# Системные зависимости для LightGBM и aiohttp
RUN apt-get update && apt-get install -y build-essential libgomp1 git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Запуск бота
CMD ["python", "main.py"]

