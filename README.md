# image-library
## 当前进度

已完成：
- 加载 OpenCLIP 预训练模型
- 单张图片向量化
- 图片相似度计算
- 素材库图片搜索
- 向量索引保存与读取
- 本地网页版：浏览、导入、以图搜图、重复组查看与清单导出

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

## 重复与近重复分组原型

```bat
python group_duplicates.py
python -m unittest test_group_duplicates -v
```

从增量索引读取向量，检查素材没有变化后生成 `data/index/duplicate_groups.json`，不删除原图，也不重新调用模型。

- exact：SHA-256 相同的完全重复文件。
- near：不同文件的向量余弦相似度至少 0.94，且 64 位差分感知哈希（dHash）距离不超过 6。两个默认值只是初始实验值，尚未经评测标定。
- similar：向量达到上述阈值，但感知哈希不通过，仅列入相似图片对，不据此合并。
- 其他图片保持独立。这些标签是算法判定，不是人工标准答案。

先把完全重复文件绑定，再按文件名排序贪心分组。不同组之间所有成员对都须满足规则，防止 A 像 B、B 像 C 就把 A/B/C 全部连起来；该策略仍受入组顺序影响，不保证最优分组。推荐保留项仅按像素数量比较，同分按文件名排序，不等同于最高画质。

调整实验参数示例（重新生成报告）：

```bat
python group_duplicates.py --similarity 0.94 --hash-distance 6
```

限制：该原型适用于小库，全部图片对的比较及结果缓存最坏为平方级开销。dHash 对裁剪、边框、水印等变化可能漏判；不能只为某张图片入组而放宽阈值，应在有来源记录的正负样本上评测。本阶段尚无人工审核页面、清理功能或完整的质量评分。

## 批量评测数据与阈值实验

```bat
python evaluate_duplicates.py prepare
python evaluate_duplicates.py evaluate
```

prepare 默认从 `data/library` 中选择 10 张 `pexels-*` 图片（随机种子 42），按原图划分为 6 张调参原图和 4 张测试原图。每张生成 7 个版本：标准化原图、文件副本、50% 缩放、亮度 70%、四边各裁去 10%、右下角水印、JPEG 质量 55。原图预处理为 RGB，最长边不超过 1024；全部文件写入独立的 `data/evaluation`，原素材不变。已存在的评测目录不会覆盖，可用 `--dataset data/evaluation_v2` 指定新目录。

自备原图目录示例：

```bat
python evaluate_duplicates.py prepare --source data/originals --pattern "*" --count 10 --dataset data/evaluation_v2
python evaluate_duplicates.py evaluate --dataset data/evaluation_v2
```

`manifest.json` 记录来源文件、哈希、变换参数和分区；需检查被选原图确为独立素材，程序只能排除文件完全重复，无法自动保证不同原图之间不是近重复。所有同源版本是正样本对，不同来源是负样本对。调参集与测试集按来源分离，不进行跨分区比较。

evaluate 建立独立增量索引，比较 9 组候选阈值（余弦 0.90/0.94/0.97 × 哈希距离 6/12/18），仅根据调参集选择候选，再与默认阈值在测试集比较。优先选择调参集查准率至少 98% 的最高召回率方案；不达标时只报告 F1 最高候选。结果写入 `evaluation_report.md` 与 `evaluation_results.json`，含 TP/FP/FN/TN、查准率、召回率、F1、误判率、错误图片对，以及原图对各变换版本的检出情况。不会自动修改分组脚本阈值。

这属于图片对判定评测，不是完整聚类评测，也不是课程要求的 15 条软件测试用例。10 张原图的结果只能用于初步诊断，不足以证明实际去重精度；若依据测试结果继续改算法，需另留新的原图用于最终验收。

```bat
python -m unittest test_evaluate_duplicates test_group_duplicates test_incremental_search -v
```

## 本地网页版

```bat
python web_app.py
```

启动后用浏览器打开提示的地址（默认 `http://127.0.0.1:8765`，可用 `--port` 修改）。服务仅监听本机，页面令牌每次启动随机生成。

- 素材库浏览：缩略图网格，支持文件名筛选、排序、按重复组合并显示。
- 导入图片：支持多选（JPG、PNG、WebP、BMP，单张上限 20 MB），同名图片自动另存，导入完成后自动更新索引。
- 以图搜图：选择或拖入查询图片即可搜索，查询图不会加入素材库。
- 重复组查看与两套阈值对比：可修改相似度与哈希距离重新分析，两套条件并排对比。
- 素材清单导出：勾选图片后导出 CSV（含尺寸、字节数与入库状态）。

网页端的索引更新采用单任务队列，任务运行期间不能进行其他导入或更新操作；关闭终端即停止服务。模型在首次搜索或处理新图时才加载，首次加载需联网下载权重。网页端调用重复分组时不写入 `duplicate_groups.json`，命令行版本的报告不受影响。
