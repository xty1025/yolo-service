# 项目结构与运行流程

## 1. 项目定位

本项目把 ESCA-Net 乳腺钼靶检测模型封装成一个可调用的 HTTP 服务：

```text
上传一张影像
    -> 返回病灶类别、置信度和边框坐标（JSON）
    -> 或返回画框后的 JPEG 图片
```

模型先在 DDSM 数据集上预训练，再迁移到医院临床数据集，当前支持 `BENIGN` 和 `MALIGNANT` 两类。部署环境是单进程、单容器、CPU 推理。

### 目标

- 验证从模型文件到在线服务的完整链路
- 展示模型加载 / 释放、接口设计、并发控制、日志和健康检查
- 让模型文件与镜像解耦，换模型不需要重建镜像

### 非目标

- 不是临床诊断产品，不处理真实患者数据
- 当前没有用户系统、数据库、任务队列和监控告警
- 当前没有完成线上 QPS、P50 / P95 压测

## 2. 系统上下文

```text
                      HTTP / multipart
  客户端  ───────────────────────────────►  FastAPI + Uvicorn
  浏览器                                        │
  curl.exe                                      ├─ 请求校验、路由
                                                ├─ OpenCV 解码 / 画框
                                                ├─ Ultralytics + ONNX Runtime
                                                ├─ Semaphore 并发控制
                                                └─ 日志落盘
                                                         │
                                                         ▼
                                              medical_best.onnx
                                              （只读挂载）
```

服务依赖一个外部模型文件，不连接数据库，也不访问其他微服务。

## 3. 目录结构

```text
yolo-service/
├── main.py                  # 应用入口：日志、lifespan、基础接口
├── ml.py                    # 模型管理：加载、获取、释放
├── routers/
│   ├── __init__.py
│   ├── files.py             # 文件上传元信息
│   └── predict.py           # 推理接口、图像解码和并发控制
├── docs/
│   ├── ARCHITECTURE.md      # 本文件
│   ├── demo_benign.jpg      # 输入样例
│   └── demo_benign_pred.jpg # 检测结果样例
├── Dockerfile               # 镜像构建
├── docker-compose.yaml      # 端口、模型挂载、环境变量
├── requirements.txt         # Python 依赖
├── README.md                # 快速开始和接口说明
└── medical_best.onnx        # 运行期模型，不入库
```

## 4. 分层与依赖

| 层 | 文件 | 职责 |
|---|---|---|
| 接入层 | `main.py` | 创建 FastAPI 实例、注册路由、配置日志和生命周期 |
| 路由层 | `routers/predict.py`、`routers/files.py` | 解析请求、组织响应 |
| 业务处理 | `routers/predict.py::run_inference()` | 解码图片、调用模型、整理检测结果 |
| 资源层 | `ml.py` | 模型单例的加载、获取与释放 |
| 基础设施 | 日志、`/health`、Docker | 可观测性和交付 |

依赖方向保持单向：

```text
main.py
   ├── routers/files.py
   └── routers/predict.py
              │
              ▼
            ml.py
```

`ml.py` 不依赖路由层，因此可以单独替换模型，或者在其他服务中复用。

## 5. 启动与关闭

启动命令：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

流程如下：

```text
导入 main.py
   ├─ 配置根日志：终端输出
   ├─ 创建 logs/ 目录，添加 RotatingFileHandler
   ├─ 导入并注册 files / predict 路由
   └─ 创建 FastAPI(lifespan=lifespan)

进入 lifespan 启动阶段
   ├─ 记录 START_TIME
   ├─ ml.load_model()
   │    └─ YOLO(MODEL_PATH) 加载 ONNX
   └─ 完成启动，开始接收请求

收到 SIGTERM / Ctrl+C
   └─ 进入 lifespan 关闭阶段
        ├─ 清空 START_TIME
        ├─ ml.unload_model()：模型引用置 None
        └─ gc.collect()
```

模型只在启动阶段加载一次。`MODEL_PATH` 优先读取环境变量；没有设置时，默认使用项目根目录下的 `medical_best.onnx`。

容器删除后，容器内 `logs/` 文件也会一起消失；当前没有把日志目录挂载到宿主机。

## 6. 请求流程

### 6.1 `POST /predict`：返回 JSON

```text
客户端
  │ multipart/form-data: file
  ▼
FastAPI 路由 predict()
  ├─ file.file.read() 得到 bytes
  └─ run_inference(bytes)
       ├─ np.frombuffer -> uint8
       ├─ cv2.imdecode -> BGR 图像
       │    └─ 失败 -> 400「不是有效图片」
       ├─ ml.get_model()
       │    └─ 未加载 -> 503「模型未就绪」
       ├─ with INFER_SLOTS
       │    └─ model.predict(conf=0.25, device='cpu', verbose=False)
       └─ 遍历 boxes / conf / cls
            └─ 组装 class_id、class_name、conf、box
  ▼
{"count": N, "detections": [...]}
```

`INFER_SLOTS` 是 `threading.Semaphore(2)`，只限制同时进入 `model.predict()` 的请求数量。

### 6.2 `POST /predict/image`：返回图片

前半段与 `/predict` 共用 `run_inference()`，之后执行：

```text
逐框绘制 cv2.rectangle
    -> cv2.putText 写类别和置信度
    -> cv2.imencode('.jpg', img)
    -> Response(..., media_type='image/jpeg')
```

### 6.3 `GET /health`

```text
ml.get_model()
    ├─ 返回模型对象 -> 200 {"status": "OK"}
    └─ RuntimeError -> 503 {"detail": "模型未就绪"}
```

它只检查模型引用是否存在，不会执行一次真实推理，因此适合做低成本健康检查。

### 6.4 其它接口

| 方法 | 路径 | 行为 |
|---|---|---|
| GET | `/` | 返回服务欢迎信息 |
| GET | `/status` | 返回 `START_TIME` 中的启动时间戳 |
| GET | `/model-info` | 返回模型路径、类别数和全部类别 |
| POST | `/files/upload` | 读取文件并返回文件名、Content-Type 和大小，不保存文件 |

## 7. 并发模型

推理接口使用普通 `def`，不是 `async def`：

1. FastAPI 会把同步路由交给线程池执行。
2. 推理本身是 CPU 阻塞任务，放在线程池里不会卡住 Uvicorn 的事件循环。
3. `Semaphore(2)` 把同时执行的 `model.predict()` 限制为 2 个。
4. 更多的同步请求仍可能占用等待线程，所以这里的 Semaphore 是推理并发上限，不是完整的请求队列或限流系统。
5. 当前 Docker 只启动 1 个 Uvicorn 进程；如果改成多 worker，每个进程都会有自己的 Semaphore，总并发会随之增加。

这套实现优先保证单机演示时的稳定性，没有针对高并发做完整设计。

## 8. 错误处理

| 状态码 | 触发条件 | 处理位置 |
|---|---|---|
| 400 | 上传内容无法被 OpenCV 解码 | `run_inference()` |
| 422 | 缺少 multipart 的 `file` 字段 | FastAPI 参数校验 |
| 503 | 模型引用不存在或已经释放 | `get_model()` 被调用处 |

服务没有统一的异常响应模型，也没有针对超大文件、恶意图片和超时做专门处理。

## 9. 日志

`main.py` 配置了两个输出位置：

1. 终端：`logging.basicConfig()`，方便直接看容器日志。
2. 文件：`logs/app_<pid>.log`，使用 `RotatingFileHandler`。

文件日志参数：

| 参数 | 值 |
|---|---|
| 单文件上限 | 5 MB |
| 备份数量 | 3 |
| 编码 | UTF-8 |

按 PID 分文件是为了避免同一镜像启动多个进程时互相覆盖。注意：文件当前写在容器内部，`docker compose down` 删除容器后不会保留。

## 10. Docker 构建与运行

构建：

```text
docker compose up -d --build
   │
   ├─ FROM python:3.12-slim
   ├─ 安装 libgl1 / libglib2.0-0（OpenCV 系统依赖）
   ├─ 从 PyTorch 官方 CPU 源安装 torch / torchvision
   ├─ 从清华源安装 requirements.txt
   └─ 只复制 main.py、ml.py 和 routers/
```

运行：

```text
docker compose up -d
   ├─ 挂载 ./medical_best.onnx -> /models/medical_best.onnx:ro
   ├─ 注入 MODEL_PATH=/models/medical_best.onnx
   ├─ 映射 8001（宿主） -> 8000（容器）
   └─ 启动 uvicorn main:app --host 0.0.0.0 --port 8000
```

模型使用只读挂载，容器不能修改宿主机上的模型文件。换模型时替换宿主文件并重启容器即可，不需要重建镜像。

## 11. ONNX 推理

### 为什么导出 ONNX

当前 `.pt` 模型包含 ESCA / DFF 等训练时注册的自定义模块。直接用普通 Ultralytics 加载可能报：

```text
ModuleNotFoundError: No module named 'ultralytics.nn.Addmodel'
```

在训练仓库中导出 ONNX 后，服务端只需要 ONNX 模型，不必携带训练仓库和自定义模块代码。

### 性能结果

测试条件：CPU、640 × 640 输入、warmup 2 次、连续 10 次取平均。包含预处理、推理和后处理。

| 后端 | 单图耗时 |
|---|---:|
| PyTorch `.pt` | 107.1 ms |
| ONNX Runtime `.onnx` | 74.6 ms |

单图耗时降低约 30.3%。这不是端到端 QPS，也不能直接外推到不同 CPU、不同图片尺寸或并发场景。

## 12. 关键取舍

| 设计 | 原因 | 代价 |
|---|---|---|
| 模块级模型单例 | 只加载一次，所有请求共享 | 模型无法在不重启进程的情况下热替换 |
| 同步路由 + 线程池 | 避免阻塞事件循环 | 线程仍会被等待中的请求占用 |
| `Semaphore(2)` | 控制 CPU 推理争抢 | 不是全局队列，也不是精确限流 |
| 模型只读挂载 | 模型和镜像解耦 | 模型文件必须由部署方单独提供 |
| 启动时加载 | 第一个业务请求不用等待模型初始化 | 启动时间变长，健康检查需要等待启动完成 |
| `/health` 只查引用 | 开销低、实现简单 | 不能发现模型损坏或推理异常 |

## 13. 现状与后续

已完成：

- JSON 和画框图片两类接口
- 模型加载、释放与单例复用
- 同步路由、线程池和 Semaphore 并发限制
- 健康检查、启动时间和模型信息接口
- 终端与文件日志、大小轮转
- Dockerfile、Docker Compose、模型只读挂载
- 医学模型接入和 ONNX Runtime CPU 推理

可以继续演进：

- 用 Pydantic 定义请求和响应模型
- 增加图片大小、格式和分辨率校验
- 用 `pytest` + `TestClient` 覆盖正常与异常路径
- 重新设计日志目录挂载或接入集中日志
- 做 P50 / P95 / QPS 压测，明确容量边界
- 增加 Nginx 反向代理、鉴权和基础监控
- 评估 GPU / TensorRT 推理和模型热更新
