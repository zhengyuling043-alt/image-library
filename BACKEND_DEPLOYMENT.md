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

## 模型来源、版本与完整性

- 框架：`open_clip_torch==3.3.0`。
- 模型：`ViT-B-32-quickgelu`，预训练权重标识 `openai`，输出512维向量。
- 权重下载地址由 OpenCLIP 官方预训练配置解析为
  `https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt`。
- 本机缓存文件：`.runtime/model-cache/ViT-B-32.pt`，大小353,976,522字节。
- SHA-256：`40D365715913C9DA98579312B702A82C18BE219CC2A73407C4526F58EBA950AF`。
- 项目代码与第三方模型边界：OpenCLIP负责预训练编码；批处理、缓存校验、设备选择、
  向量索引、增量维护、去重判定、聚类、解释和API由本项目实现。

## 可复现依赖

`requirements.txt`给出兼容范围，`requirements-lock.txt`记录本次验收机器的精确版本。
CI使用Python 3.12运行不下载模型权重的54项单元与接口测试；真实模型冒烟与性能测试在
本机D盘缓存上执行。

```powershell
./.venv/Scripts/python.exe -m unittest discover -v
./.venv/Scripts/python.exe benchmark_model_batch.py --images 8 --batch-size 8
./.venv/Scripts/python.exe benchmark_pipeline.py
```

实测结果分别见 `MODEL_BATCH_BENCHMARK_RESULTS.md`、`PIPELINE_BENCHMARK_RESULTS.md`
和 `LIBRARY_EVALUATION_RESULTS.md`。不同硬件结果会变化，不对未实测规模外推模型耗时。
