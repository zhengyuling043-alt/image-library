# B 后端与模型部署说明

## 运行环境

- 推荐 Python 3.12；项目虚拟环境固定在 `.venv`。
- pip 下载缓存与 OpenCLIP 权重固定在 `.runtime`，不会写入 Git。
- 默认自动选择 CUDA；没有可用 GPU 时使用 CPU。可通过
  `IMAGE_LIBRARY_DEVICE=cpu` 强制使用 CPU。

## 安装与启动

```powershell
Set-Location "D:\master-1 paper\image-library-backend"
.\setup_backend.ps1
.\start_backend.ps1
```

浏览器访问 `http://127.0.0.1:8765`。健康检查地址为
`http://127.0.0.1:8765/api/health`，可查看 Python、模型、设备、向量维度和
索引状态。健康检查不会为了展示状态而加载大模型。

## 模型边界与个人实现

OpenCLIP 已有能力是将图片编码为 512 维语义向量。本分支负责的工程能力包括：
项目内模型缓存、CPU/GPU 自动选择、离线复用、运行状态接口、增量索引、
重复判定与后端任务接口。模型本身的预训练不属于个人实现。
