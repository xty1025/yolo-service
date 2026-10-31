FROM python:3.12-slim

WORKDIR /app

# 系统库：opencv 在 Linux 上缺这两个会 import 失败
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 第一步：torch 从官方 CPU 源装
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 第二步：其余依赖从清华源装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 只拷运行需要的文件，别 COPY . 全塞进去
COPY main.py ml.py ./
COPY routers ./routers

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]