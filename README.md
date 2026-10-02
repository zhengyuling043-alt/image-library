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
近重复分组和增量更新尚未实现。