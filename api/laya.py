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
    优先读取环境变量MODEL_PATH，若本地路径存在直接加载；
    模型ref仅作为标识，离线场景忽略huggingface下载
    """
    MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    # 离线校验模型文件
    config_file = os.path.join(MODEL_PATH, "config.json")
    if not os.path.exists(config_file):
        raise RuntimeError(f"Model missing at {MODEL_PATH}, expect config.json. ref={model_ref}")

    return LayaAgent(MODEL_PATH, DEVICE, TORCH_DTYPE)
