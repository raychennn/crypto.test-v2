FROM python:3.11-slim

WORKDIR /app

# 防止 Python 產生 .pyc 檔案 / 確保 log 即時輸出
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
