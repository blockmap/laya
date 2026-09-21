FROM pytorch/pytorch:2.3.0-cuda12.0-cudnn8-runtime

WORKDIR /app

# 1. 安装依赖
COPY requirements.txt .
# 在线构建模式：build机有外网，直接pip安装
RUN pip install --no-cache-dir -r requirements.txt

# 2. 构建阶段下载模型到镜像内 /app/model
# 安装huggingface-hub用于拉取模型
RUN pip install --no-cache-dir huggingface-hub
ENV HF_HUB_ENABLE_HF_TRANSFER=1
# 拉取ConvAI/Laya，保存到 /app/model
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ConvAI/Laya', local_dir='/app/model', local_dir_use_symlinks=False)"

# 3. 拷贝业务代码
COPY main.py .

# 环境变量：推理时读取内置模型，强制离线加载
ENV MODEL_PATH=/app/model
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python", "main.py"]
