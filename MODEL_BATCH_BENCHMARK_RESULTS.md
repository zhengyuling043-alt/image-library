# OpenCLIP真实批处理测试

缓存权重后的真实OpenCLIP CPU/GPU推理；包含图片解码与预处理，不包含下载。

- 模型：`ViT-B-32-quickgelu` / `openai`；设备：`cpu`；样本：8 张。
- 模型加载：17.959 s（权重已缓存于D盘）。
- 逐张编码：0.657 s，12.169 张/s。
- 批量编码：0.567 s，14.105 张/s。
- 本机实测加速比：1.159x；RSS变化：927.93 MB。

命令：`.venv\Scripts\python.exe benchmark_model_batch.py --images 8 --batch-size 8`。
不同CPU/GPU和冷启动状态会改变结果；报告不外推未实测的5000张模型耗时。
