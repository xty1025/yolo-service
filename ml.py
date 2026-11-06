from pathlib import Path
from  ultralytics import YOLO
import logging
import gc
import os
logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "medical_best.onnx"))
_model=None
def load_model():
    global _model
    logger.info(f">>> 加载模型中：{MODEL_PATH.name}")
    _model=YOLO(str(MODEL_PATH))
    logger.info("模型加载完成")
    return _model
def get_model():
    "取模型"
    if _model is None:
        raise RuntimeError("模型没有加载好")
    return _model

def unload_model():
    global _model
    logger.info(">>>释放模型")
    _model=None
    gc.collect()
    logger.info(">>>模型引用已释放")
