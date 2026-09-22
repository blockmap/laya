# laya.py
import os
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any


class LayaAgent:
    def __init__(self, model_path: str, device: str, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        print(f"Loading Laya from {model_path}, device={self.device}")
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch_dtype,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.model.eval()
        print("Laya agent loaded ready.")

    @torch.no_grad()
    def predict(self, state_dict: Dict[str, Any], questions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Laya官方风格predict接口
        :param state_dict: dict，例如 {"body": "文本内容"}，内部拼接为state字符串
        :param questions: jev格式questions定义
        :return: {"answers": {...}} 完全兼容Jev输出
        """
        # 将state dict拼接为state文本（和官方行为对齐，key:value换行）
        state = "\n".join([f"{k}: {v}" for k, v in state_dict.items()])
        answers = {}

        for qid, qdef in questions.items():
            q_type = qdef["type"]
            instr = qdef["instructions"]

            if q_type == "noul":
                prompt = f"{state}{self.tokenizer.sep_token}{instr}"
                inputs = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt"
                ).to(self.device)
                out = self.model(**inputs)
                cls_emb = out.last_hidden_state[:, 0, :]
                # 此处替换为Laya原生打分头，当前sigmoid作为占位
                logit = torch.nn.Linear(cls_emb.shape[-1], 1).to(self.device)(cls_emb)
                score = torch.sigmoid(logit).item()
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(score), 4)
                }

            elif q_type == "choice":
                criteria = qdef["criteria"]
                opt_keys = list(criteria.keys())
                opt_scores = {}
                for opt_key, opt_text in criteria.items():
                    prompt = f"{state}{self.tokenizer.sep_token}{instr}{self.tokenizer.sep_token}{opt_text}"
                    inputs = self.tokenizer(
                        prompt,
                        truncation=True,
                        max_length=8192,
                        return_tensors="pt"
                    ).to(self.device)
                    out = self.model(**inputs)
                    cls_emb = out.last_hidden_state[:, 0, :]
                    raw_logit = torch.nn.Linear(cls_emb.shape[-1], 1).to(self.device)(cls_emb)
                    opt_scores[opt_key] = raw_logit.item()
                # softmax概率
                keys = list(opt_scores.keys())
                vals = torch.tensor([opt_scores[k] for k in keys])
                probs = torch.softmax(vals, dim=0)
                prob_dict = {keys[i]: round(float(probs[i]), 4) for i in range(len(keys))}
                best_key = max(prob_dict, key=prob_dict.get)
                answers[qid] = {
                    "type": "choice",
                    "choice": best_key,
                    "confidence": round(prob_dict[best_key].item(), 4),
                    "probabilities": prob_dict
                }

            elif q_type == "score":
                criteria = qdef["criteria"]
                opt_scores = {}
                for idx, opt_text in enumerate(criteria):
                    prompt = f"{state}{self.tokenizer.sep_token}{instr}{self.tokenizer.sep_token}{opt_text}"
                    inputs = self.tokenizer(
                        prompt,
                        truncation=True,
                        max_length=8192,
                        return_tensors="pt"
                    ).to(self.device)
                    out = self.model(**inputs)
                    cls_emb = out.last_hidden_state[:, 0, :]
                    raw_logit = torch.nn.Linear(cls_emb.shape[-1], 1).to(self.device)(cls_emb)
                    opt_scores[str(idx)] = raw_logit.item()
                keys = list(opt_scores.keys())
                vals = torch.tensor([opt_scores[k] for k in keys])
                probs = torch.softmax(vals, dim=0)
                prob_dict = {keys[i]: round(float(probs[i]), 4) for i in range(len(keys))}
                total_score = sum(int(k) * prob_dict[k] for k in prob_dict)
                answers[qid] = {
                    "type": "score",
                    "score": round(total_score, 4),
                    "confidence": round(max(prob_dict.values()), 4),
                    "probabilities": prob_dict
                }
        return {"answers": answers}


def load(model_ref: str) -> LayaAgent:
    """
    laya.load("convaiinnovations/laya-multilingual")
    优先加载本地MODEL_PATH；若模型缺失则自动从HuggingFace下载；
    下载失败（或HF_HUB_OFFLINE=1）时抛出带修复指引的RuntimeError
    """
    MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    config_file = os.path.join(MODEL_PATH, "config.json")
    if not os.path.exists(config_file):
        _download_model(model_ref, MODEL_PATH, config_file)

    return LayaAgent(MODEL_PATH, DEVICE, TORCH_DTYPE)


def _download_model(model_ref: str, model_path: str, config_file: str) -> None:
    """模型本地缺失时从HuggingFace下载到MODEL_PATH，并给出可执行的修复指引"""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError(_missing_model_msg(model_ref, model_path, "HF_HUB_OFFLINE=1 已设置（离线模式），不会尝试联网下载。"))

    try:
        from huggingface_hub import snapshot_download

        os.makedirs(model_path, exist_ok=True)
        snapshot_download(
            repo_id=model_ref,
            local_dir=model_path,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as e:
        raise RuntimeError(_missing_model_msg(model_ref, model_path, f"自动下载失败: {e}")) from e

    if not os.path.exists(config_file):
        raise RuntimeError(
            f"Downloaded {model_ref} to {model_path}, but {config_file} still missing. "
            "The repo layout may not match a transformers model."
        )


def _missing_model_msg(model_ref: str, model_path: str, reason: str) -> str:
    return (
        f"Model missing at {model_path}, expect config.json. ref={model_ref}. {reason}\n"
        "Fix options:\n"
        "  1) 给容器联网权限，启动时自动从HuggingFace下载（公开模型无需HF_TOKEN）\n"
        "  2) 私有仓库模型：设置环境变量 HF_TOKEN 后重启\n"
        "  3) 离线部署：宿主机先把模型放到 ./model/，并取消docker-compose.yaml中 "
        "volumes 的 `- ./model:/app/model` 注释\n"
        "  4) 构建期烘焙模型到镜像：docker build --build-arg DOWNLOAD_MODEL=true"
    )
