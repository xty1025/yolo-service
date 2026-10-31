# 项目结构与运行流程

## 1. 项目定位

一个把目标检测模型封装成 HTTP 服务的工程实践项目。
输入一张图片，输出检测结果（JSON）或画框后的图片；整个服务可打包成 Docker 镜像交付。

## 2. 目录结构

```
yolo-service/
├── main.py               # 应用入口：日志配置、生命周期、基础接口
├── ml.py                 # 模型管理：加载 / 获取 / 释放
├── routers/
│   ├── __init__.py
│   ├── files.py          # 文件上传（通用工具接口）
│   └── predict.py        # 推理接口（业务核心）
├── docs/
│   └── ARCHITECTURE.md   # 本文件
├── Dockerfile            # 镜像定义
├── docker-compose.yaml   # 编排：端口、模型挂载、环境变量
├── requirements.txt      # 依赖清单
├── README.md             # 快速开始
└── yolo26n.onnx          # 运行期模型（不入库）
```

## 3. 分层与职责

| 层 | 文件 | 职责 |
|---|---|---|
| 接入层 | `main.py` | 创建 FastAPI 实例、注册路由、应用级配置 |
| 路由层 | `routers/predict.py`、`routers/files.py` | 解析请求、组织响应 |
| 业务层 | `predict.py: run_inference()` | 字节解码、调用模型、结果整理 |
| 资源层 | `ml.py` | 模型单例：加载、获取、释放 |
| 基础设施 | 日志、`/health`、Docker | 可观测性与交付 |

依赖方向是单向的：`main → routers → ml`。`ml.py` 不反向依赖任何路由，所以它可以被单独测试或替换。

## 4. 启动流程

```
uvicorn main:app
   │
   ├─ 导入 main.py：执行模块顶层代码
   │    ├─ 配置日志（终端 + 文件）
   │    ├─ 导入 routers，注册路由
   │    └─ 创建 app = FastAPI(lifespan=...)
   │
   ├─ 进入 lifespan（启动阶段）
   │    ├─ START_TIME['at'] = time.time()
   │    ├─ ml.load_model() → YOLO(MODEL_PATH)
   │    └─ 日志 "准备完了"
   │
   ├─ 开始接收请求 ────────────────┐
   │                               │
   └─ 收到停止信号（Ctrl+C / docker stop）
        └─ 退出 lifespan（关闭阶段）
             ├─ START_TIME.clear()
             └─ ml.unload_model() → 置 None → gc → cuda.empty_cache()
```

模型只在启动时加载一次，之后所有请求共用这一份，避免每次推理都读盘。

## 5. 请求流程

### 5.1 POST /predict（返回 JSON）

```
客户端                    服务
  │  multipart/form-data    │
  ├────────────────────────►│ uvicorn 解析出 UploadFile
  │                         │ 路由匹配 → routers/predict.py :: predict()
  │                         │ file.file.read() → bytes
  │                         │ run_inference(bytes)
  │                         │   ├─ np.frombuffer → uint8 数组
  │                         │   ├─ cv2.imdecode → BGR 图像（失败 → 400）
  │                         │   ├─ ml.get_model()（None → 503）
  │                         │   └─ with GPU_LIMIT: model.predict(conf=0.25)
  │                         │ zip(boxes, conf, cls) → 结果列表
  │◄────────────────────────┤ {"count": 5, "detections": [...]}
```

### 5.2 POST /predict/image（返回图片）

前 6 步与上面完全一致（共用 `run_inference`），之后：

```
逐框 cv2.rectangle + cv2.putText
   → cv2.imencode('.jpg', img)
   → Response(content=..., media_type="image/jpeg")
```

### 5.3 GET /health

```
ml.get_model()  ── 成功 ──► 200 {"status": "OK"}
                └─ RuntimeError ──► 503 {"detail": "模型未就绪"}
```

用于容器编排判断"服务是否真的可用"，而不是"进程是否还活着"。

### 5.4 其它接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 欢迎信息 |
| GET | `/status` | 进程启动时间 |
| GET | `/model-info` | 模型路径、类别数、前 3 个类别 |
| POST | `/files/upload` | 上传并返回文件元信息 |

## 6. 部署流程

```
docker compose up -d --build
   │
   ├─ build：读取 Dockerfile
   │    ├─ FROM python:3.12-slim
   │    ├─ apt 安装 libgl1 / libglib2.0-0（OpenCV 依赖）
   │    ├─ 安装 torch CPU 版（官方源）
   │    ├─ 安装 requirements.txt（清华源）
   │    └─ 拷贝 main.py / ml.py / routers
   │
   └─ run：
        ├─ 挂载 yolo26n.onnx → /models/yolo26n.onnx（只读）
        ├─ 注入 MODEL_PATH=/models/yolo26n.onnx
        └─ 端口 8001（宿主） → 8000（容器）
```

模型和配置都在镜像外，换模型不需要重建镜像。

## 7. 关键设计取舍

| 设计 | 原因 |
|---|---|
| 模型做成模块级单例 | 加载一次，全进程共享，避免重复占内存 |
| 路由函数用 `def` 而非 `async def` | FastAPI 会把 `def` 放进线程池，阻塞的推理不会卡住事件循环 |
| `Semaphore(2)` 限制并发 | 推理是 CPU/GPU 密集型，放太多并发只会互相拖慢、挤爆显存 |
| 日志双出口 | 终端方便调试，文件用于事后排查；按 PID 分文件避免多进程互相覆盖 |
| 模型走挂载而非打进镜像 | 换模型不必重建镜像，镜像也不因模型变大 |
| `/health` 检查模型而非进程 | 进程活着但模型没加载，同样不能接客 |

## 8. 现状与后续

已完成：接口、并发控制、日志、健康检查、优雅关闭、容器化、ONNX 推理。

可继续演进的方向：

- 用 Pydantic 定义请求/响应模型，让接口契约显式化
- 上传大小与类型校验
- `pytest` + `TestClient` 覆盖主要路径
- 增加前端上传页面
- GPU 版本镜像与 TensorRT 加速