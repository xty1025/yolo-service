# YOLO 目标检测模型服务化部署

基于 YOLO + FastAPI 的目标检测推理服务：上传一张图片，返回检测结果 JSON 或画好框的可视化图像。当前示例模型为 COCO，可通过 `MODEL_PATH` 替换为自研医学影像模型。
服务已容器化，可通过 Docker 一键复现。

## 功能

- `POST /predict`：返回检测结果 JSON（类别、置信度、边框坐标）
- `POST /predict/image`：返回画框后的 JPEG 图像
- `GET /health`：健康检查（200 正常 / 503 模型未就绪）
- `GET /status`：服务启动时间
- `GET /model-info`：当前模型信息
- 推理接口使用同步路由，由 FastAPI 放入线程池执行，并用 `Semaphore` 限制单进程并发，避免推理资源争抢
- 日志按进程号落盘并自动轮转，支持优雅关闭时释放模型

## 技术栈

| 层次 | 选型 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| 推理 | PyTorch / Ultralytics，ONNX Runtime |
| 图像处理 | OpenCV |
| 部署 | Docker、Docker Compose |

## 快速开始

### 1. 准备模型

仓库不包含模型文件，请自行导出（需要已安装 ultralytics）：

```bash
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt').export(format='onnx', imgsz=640)"
```

将生成的 `yolo26n.onnx` 放到本目录下（`docker-compose.yaml` 默认挂载这个路径）。

> 也可以直接使用 `.pt` 模型，只需把 compose 里的 `MODEL_PATH` 和挂载路径改成 `yolo26n.pt`。

### 2. 启动服务

```bash
docker compose up -d --build
curl http://127.0.0.1:8001/health
```

### 3. 调用

```bash
curl -X POST -F "file=@test.jpg" http://127.0.0.1:8001/predict
curl -X POST -F "file=@test.jpg" -o out.jpg http://127.0.0.1:8001/predict/image
```

接口文档：<http://127.0.0.1:8001/docs>

## 本地开发

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 上传图片，返回检测结果 JSON |
| POST | `/predict/image` | 上传图片，返回画框图像 |
| GET | `/health` | 健康检查 |
| GET | `/status` | 启动时间 |
| GET | `/model-info` | 模型信息 |

## 性能对比（CPU，640×640，10 次平均）

| 推理后端 | 单图耗时 |
|---|---|
| PyTorch (.pt) | 34.2 ms |
| ONNX Runtime (.onnx) | 25.1 ms |

导出 ONNX 后单图耗时下降约 26.6%，吞吐提升约 36%。

## 目录结构

```
yolo-service/
├── main.py               # 应用入口：lifespan、日志、健康检查
├── ml.py                 # 模型加载 / 获取 / 释放
├── routers/
│   ├── files.py          # 文件上传
│   └── predict.py        # 推理接口
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── yolo26n.onnx          # 模型文件（不入库）
```

## 更多文档

- [项目结构与运行流程](docs/ARCHITECTURE.md)

## 说明

- 模型与 `.env` 均不纳入版本控制，通过挂载和环境变量注入
- 当前仅提供 CPU 部署；GPU 版本需要额外替换 ONNX Runtime、修改推理设备配置，并准备 CUDA/cuDNN 环境，暂未包含
