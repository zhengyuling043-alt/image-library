# image-library
## 当前进度

已完成：
- 加载 OpenCLIP 预训练模型
- 单张图片向量化
- 图片相似度计算
- 素材库图片搜索
- 向量索引保存与读取

当前模型：ViT-B-32-quickgelu
预训练权重：openai
运行方式：CPU

## 环境安装

本项目当前使用 Python 3.14。

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

以上激活命令适用于 Windows cmd 终端。
首次加载模型需要联网下载权重。

## 测试数据

测试图片和生成的索引不包含在仓库中，需要自行准备：
- data/library/：放置素材图片
- data/test.jpg：默认查询图片
- data/test_other.jpg：另一张查询图片

## 运行方式

建立索引：

```bat
python indexed_search.py build
```

搜索：

```bat
python indexed_search.py search
```

更换查询图片：

```bat
python indexed_search.py search --query "data/test_other.jpg"
```

当前使用 NumPy 保存向量并进行全量相似度比较。
素材新增、修改或删除后，暂时需要重新建立索引。
以上为 indexed_search.py 基础版本的行为。近重复分组尚未实现。

## 增量更新版本

`incremental_search.py` 支持新增、修改、删除的索引同步。
首次运行会为素材建立新的增量缓存，原有 vectors.npy 和 metadata.json 保留。

```bat
python incremental_search.py update
python incremental_search.py search
python incremental_search.py search --query "data/test_other.jpg"
```

每次素材变化后运行 update。程序读取文件内容并计算 SHA-256，只有新图或内容变化的图片才调用模型；未变化的图片复用向量，删除的图片从索引移除。没有变化时不会加载模型。

增量索引保存于 `data/index/incremental.npz`，包含向量、文件名、哈希和模型配置，以单文件原子替换避免向量与元数据保存不一致。运行结果及失败原因保存于 `data/index/incremental.report.json`。失败图片不会进入搜索结果，下次 update 会重试；被修改后处理失败的图片也不会保留旧向量供搜索。

本版本仍采用 NumPy 全量向量比较，适用于当前小规模原型；文件扫描和哈希计算仍需读取全部图片。更新中断后，重新执行会从上次完整保存的索引继续，未保存的新增向量需重新计算。请勿同时运行多个更新进程。

若需要重新编码全部素材：

```bat
python incremental_search.py update --rebuild
```

索引逻辑测试（使用替代编码器，不下载模型）：

```bat
python -m unittest test_incremental_search -v
```
