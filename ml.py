from pathlib import Path
from  ultralytics import YOLO
import logging
import torch
import gc
import os
logger = logging.getLogger(__name__)
MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "yolo26n.onnx"))
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

def mb(x):
    return x / 2**20
def unload_model():
    global _model
    logger.info(">>>释放模型")
    logger.info(f"[前] allocated={mb(torch.cuda.memory_allocated()):.0f}MiB "
                f"reserved={mb(torch.cuda.memory_reserved()):.0f}MiB")
    _model=None
    gc.collect()
    logger.info(f"[后] allocated={mb(torch.cuda.memory_allocated()):.0f}MiB "
                f"reserved={mb(torch.cuda.memory_reserved()):.0f}MiB")
    torch.cuda.empty_cache()
    logger.info(">>>显存已经归还")
