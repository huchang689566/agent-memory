FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt sentence-transformers

COPY . .

RUN mkdir -p /app/data /app/data/sessions

EXPOSE 8001 8002
