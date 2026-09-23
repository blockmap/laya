# CUDA12.0.1 + cudnn8，ubuntu22.04
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 安装 python3/pip 与 compose healthcheck 所需的 curl（单层安装并清理 apt 缓存）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

# 依赖唯一来源 requirements.txt（torch==2.5.1 等）：升级 pip + 安装全部依赖合并为一个 RUN，整份 Dockerfile 仅此一次 pip install
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# 构建期下载相关 ENV/ARG（集中在此，requirements 层缓存保持稳定）
# hf_transfer 加速需要该开关；HF_TOKEN 仅作为构建参数透传（私有仓库需要），不写入镜像 ENV
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ARG HF_TOKEN
# 构建期烘焙模型：默认 false 保持镜像轻量；运行时 /app/model/encoder/config.json 缺失时 laya.py 会自动回退下载
# 需要烘焙时显式开启：docker build --build-arg DOWNLOAD_MODEL=true
ARG DOWNLOAD_MODEL=false
ARG MODEL_REPO=convaiinnovations/laya-multilingual

# 条件下载：true 时用 huggingface-hub 现代 CLI `hf download` 下载到 /app/model；false 时仅创建空目录
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
      hf download "$MODEL_REPO" --local-dir /app/model ${HF_TOKEN:+--token "$HF_TOKEN"}; \
    else \
      mkdir -p /app/model; \
    fi

# 拷贝代码：laya模块 + main + test脚本（放在依赖/模型之后，代码变更不触发重装依赖）
COPY laya.py .
COPY main.py .
COPY test_laya.py .

ENV MODEL_PATH=/app/model
ENV API_KEY=sk-laya-local-001
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python3", "main.py"]
