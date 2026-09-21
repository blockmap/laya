# main.py
import os
import torch
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union
from transformers import AutoModel, AutoTokenizer

# ===================== 配置 =====================
app = FastAPI(title="Laya Jev-Compatible API")
API_KEY = os.getenv("API_KEY", "sk-laya-local-001")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ===================== 加载Laya模型（一次性全局加载，离线模式） =====================
print(f"Loading Laya model from {MODEL_PATH}, device={DEVICE}, dtype={TORCH_DTYPE}")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=TORCH_DTYPE,
    device_map=DEVICE
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)
model.eval()
print("Model loaded successfully.")

# ===================== Pydantic Schema，严格对齐Jev /v1/systemone 请求 =====================
class NoulQuestion(BaseModel):
    type: str = "noul"
    instructions: str

class ChoiceQuestion(BaseModel):
    type: str = "choice"
    instructions: str
    criteria: Dict[str, str]

class ScoreQuestion(BaseModel):
    type: str = "score"
    instructions: str
    criteria: list[str]

Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]

class SystemOneRequest(BaseModel):
    model: str
    state: str
    questions: Dict[str, Question]

# ===================== Laya 推理核心实现 =====================
@torch.no_grad()
def laya_evaluate(state: str, questions: Dict[str, dict]) -> Dict[str, Any]:
    """
    Laya 并行评估多question，输出严格符合Jev返回结构
    Laya是双向encoder，输入格式参考ConvAI Laya规范：
    样本格式：<state> [SEP] <question instruction> [SEP] option_text
    对choice/score，遍历所有候选选项打分；noul直接输出0~1分值
    """
    answers = {}
    for qid, qdef in questions.items():
        q_type = qdef["type"]
        instr = qdef["instructions"]

        if q_type == "noul":
            # noul：0~1标量分值
            prompt = f"{state}{tokenizer.sep_token}{instr}"
            inputs = tokenizer(
                prompt,
                truncation=True,
                max_length=8192,
                return_tensors="pt"
            ).to(DEVICE)
            out = model(**inputs)
            # Laya取CLS向量做二分类分值，映射0~1
            cls_emb = out.last_hidden_state[:, 0, :]
            score = torch.sigmoid(torch.nn.Linear(cls_emb.shape[-1],1).to(DEVICE)(cls_emb)).item()
            answers[qid] = {
                "type": "noul",
                "noul": round(float(score),4)
            }

        elif q_type == "choice":
            # choice：遍历所有选项，计算每个选项匹配分数
            criteria = qdef["criteria"]
            opt_keys = list(criteria.keys())
            opt_scores = {}
            for opt_key, opt_text in criteria.items():
                prompt = f"{state}{tokenizer.sep_token}{instr}{tokenizer.sep_token}{opt_text}"
                inputs = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt"
                ).to(DEVICE)
                out = model(**inputs)
                cls_emb = out.last_hidden_state[:,0,:]
                raw = torch.nn.Linear(cls_emb.shape[-1],1).to(DEVICE)(cls_emb)
                opt_scores[opt_key] = raw.item()
            # softmax转概率分布
            keys = list(opt_scores.keys())
            vals = torch.tensor([opt_scores[k] for k in keys])
            probs = torch.softmax(vals, dim=0)
            prob_dict = {keys[i]: round(float(probs[i]),4) for i in range(len(keys))}
            best_key = max(prob_dict, key=prob_dict.get)
            best_conf = prob_dict[best_key]
            answers[qid] = {
                "type": "choice",
                "choice": best_key,
                "confidence": round(best_conf,4),
                "probabilities": prob_dict
            }

        elif q_type == "score":
            # score：criteria是有序等级列表，遍历打分，softmax概率
            criteria = qdef["criteria"]
            opt_scores = {}
            for idx, opt_text in enumerate(criteria):
                prompt = f"{state}{tokenizer.sep_token}{instr}{tokenizer.sep_token}{opt_text}"
                inputs = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt"
                ).to(DEVICE)
                out = model(**inputs)
                cls_emb = out.last_hidden_state[:,0,:]
                raw = torch.nn.Linear(cls_emb.shape[-1],1).to(DEVICE)(cls_emb)
                opt_scores[str(idx)] = raw.item()
            keys = list(opt_scores.keys())
            vals = torch.tensor([opt_scores[k] for k in keys])
            probs = torch.softmax(vals, dim=0)
            prob_dict = {keys[i]: round(float(probs[i]),4) for i in range(len(keys))}
            # 加权求score
            total_score = 0.0
            for k,p in prob_dict.items():
                total_score += int(k)*p
            answers[qid] = {
                "type": "score",
                "score": round(total_score,4),
                "confidence": round(max(prob_dict.values()),4),
                "probabilities": prob_dict
            }
    return answers

# ===================== API接口 =====================
@app.post("/v1/systemone")
async def systemone(
    req: SystemOneRequest,
    authorization: Optional[str] = Header(None)
):
    # API Key鉴权
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        answers = laya_evaluate(req.state, req.questions.model_dump())
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
