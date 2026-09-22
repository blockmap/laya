# CUDA12.0.1 + cudnn8，ubuntu22.04
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 安装python/pip基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 升级pip
RUN python3 -m pip install --upgrade pip

# 复制requirements
COPY requirements.txt .

# 安装pytorch 2.3.0 cu121 + 其他包
RUN pip install --no-cache-dir -r requirements.txt

# 安装huggingface-hub + hf_transfer（HF_HUB_ENABLE_HF_TRANSFER 生效需要 hf_transfer）
RUN pip install --no-cache-dir huggingface-hub hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ARG HF_TOKEN

# 构建期烘焙模型（默认关闭保持镜像轻量；运行时 laya.py 会自动回退下载）
ARG DOWNLOAD_MODEL=false
ARG MODEL_REPO=convaiinnovations/laya-multilingual
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
      python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ['MODEL_REPO'], local_dir='/app/model', token=os.environ.get('HF_TOKEN'))"; \
    else \
      mkdir -p /app/model; \
    fi

# 拷贝代码：laya模块 + main + test脚本
COPY laya.py .
COPY main.py .
COPY test_laya.py .

ENV MODEL_PATH=/app/model
ENV API_KEY=sk-laya-local-001
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python3", "main.py"]
