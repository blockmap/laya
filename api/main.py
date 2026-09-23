# main.py
import logging
import os
import sys
import time
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union
import laya

# ---------------- Logging（stdlib，仅受 LOG_LEVEL 环境变量控制）----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("laya.api")

app = FastAPI(title="Laya Jev-Compatible API")
API_KEY = os.getenv("API_KEY", "sk-laya-local-001")
MODEL_REF = "convaiinnovations/laya-multilingual"

# 模块加载即输出启动信息（绝不打印 key 的值）
logger.info(
    "starting: model=%s MODEL_PATH=%s api_key_configured=%s",
    MODEL_REF, os.getenv("MODEL_PATH", "<unset>"), "yes" if API_KEY else "no",
)

# 全局加载 agent，服务启动一次性加载
load_started = time.perf_counter()
try:
    agent = laya.load(MODEL_REF)
except Exception:
    logger.exception("Failed to load Laya model: %s", MODEL_REF)
    sys.exit(1)
logger.info(
    "Laya model loaded: load_seconds=%.2f device=%s",
    time.perf_counter() - load_started, agent.device,
)

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
        reason = (
            "missing Authorization header"
            if authorization is None
            else "malformed Authorization header (expected 'Bearer ' prefix)"
        )
        logger.warning("Auth rejected: %s", reason)
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        logger.warning("Auth rejected: invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # 适配：API的state字符串转state_dict {"body": state文本}
        state_dict = {"body": req.state}
        started = time.perf_counter()
        result = agent.predict(state_dict, req.questions.model_dump())
        logger.info(
            "systemone ok: questions=%d state_chars=%d duration_ms=%.1f",
            len(req.questions), len(req.state),
            (time.perf_counter() - started) * 1000,
        )
        return result
    except Exception as e:
        logger.exception("systemone inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
async def health():
    # docker-compose healthcheck 每 30s 轮询，INFO 会刷屏，仅 DEBUG 输出
    logger.debug("health check ok")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # log_config=None：uvicorn 不自带日志配置，其日志器向上冒泡到本模块的 root 配置
    uvicorn.run(
        "main:app", host="0.0.0.0", port=8000,
        log_config=None, log_level=LOG_LEVEL.lower(),
    )
