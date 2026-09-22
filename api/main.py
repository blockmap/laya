# main.py
import os
import sys
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union
import laya

app = FastAPI(title="Laya Jev-Compatible API")
API_KEY = os.getenv("API_KEY", "sk-laya-local-001")

# 全局加载 agent，服务启动一次性加载
try:
    agent = laya.load("convaiinnovations/laya-multilingual")
except Exception as e:
    print(f"[FATAL] Failed to load Laya model: {e}", file=sys.stderr)
    sys.exit(1)

# ---------------- Pydantic Schema ----------------
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

# ---------------- Jev API ----------------
@app.post("/v1/systemone")
async def systemone(
    req: SystemOneRequest,
    authorization: Optional[str] = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # 适配：API的state字符串转state_dict {"body": state文本}
        state_dict = {"body": req.state}
        result = agent.predict(state_dict, req.questions.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
