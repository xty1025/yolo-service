
from fastapi import FastAPI,HTTPException
from routers import files,predict
import time
from contextlib import asynccontextmanager
import ml
import logging
import  os
from pathlib import Path
from logging.handlers import RotatingFileHandler

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
)

LOG_DIR=Path("logs")
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"app_{os.getpid()}.log"

file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
))
logging.getLogger().addHandler(file_handler)   # ← 后


logger = logging.getLogger(__name__)
START_TIME = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(">>[启动]")
    START_TIME['at'] = time.time()#记录时间
    ml.load_model()
    logger.info(">>>[启动]准备完了")
    yield
    logger.info(">>>[关闭]")
    START_TIME.clear()
    ml.unload_model()
app = FastAPI(lifespan=lifespan)#告诉FastAPI使用lifespan
app.include_router(files.router)
app.include_router(predict.router)
@app.get("/")
def home():
    return {"message": "你好"}
@app.get("/status")
def status():
    return {"started_at": START_TIME['at']}

@app.get("/model-info")
def model_info():
    model=ml.get_model()
    names = model.names
    return {
        "model_path": str(ml.MODEL_PATH),
        "class_count": len(names),
        "first_3_classes": [names[i] for i in range(3)],
    }
@app.get("/health")
def health():
    try:
        ml.get_model()
    except RuntimeError:
        raise HTTPException(status_code=503,detail="模型未就绪")
    return {"status": "OK"}
