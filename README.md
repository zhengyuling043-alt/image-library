# image-library
## 当前进度

已完成：
- 加载 OpenCLIP 预训练模型
- 单张图片向量化
- 图片相似度计算
- 素材库图片搜索
- 向量索引保存与读取
- 本地网页版：浏览、导入、以图搜图、重复组查看与清单导出
- 57 项自动化测试全绿；四路信号消融实验与人工真值交叉验证

当前模型：ViT-B-32-quickgelu
预训练权重：openai
运行方式：自动选择 CUDA，GPU 不可用时使用 CPU

## 后端工程化版本

`backend/b-model-index` 分支在原型功能之上新增：

- Python 3.12 独立环境，依赖、pip缓存和模型权重均存放在D盘项目目录。
- OpenCLIP官方权重离线复用、CPU/GPU自动选择和 `/api/health` 诊断接口。
- NPZ兼容存储 + FAISS余弦检索侧车 + SQLite稳定图片ID和状态目录。
- 每25张写入一次原子检查点，意外中断后复用已经编码的向量继续入库。
- SHA-256、dHash和CLIP三档判定、全连接分组约束与medoid代表图。
- 分辨率、清晰度、曝光综合质量评分及可解释的保留推荐。
- 图片对指标和聚类级Precision、Recall、F1、纯度、完整组召回率。
- 分页目录API，以及包含哈希、模型版本、分组和质量分的CSV/JSON导出。
- 500/2000/5000向量检索基准，见 `BENCHMARK_RESULTS.md`。

## 环境安装

本项目推荐使用 Python 3.12。PowerShell中直接运行以下脚本即可把虚拟环境和
下载缓存建立在当前D盘项目目录：

```powershell
.\setup_backend.ps1
.\start_backend.ps1
```

也可手动执行：

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

以上激活命令适用于 Windows cmd 终端。
首次加载模型需要联网下载权重。

## 测试数据

仓库附带课程演示数据：108 个原图来源及其变体，共 500 张图片，以及对应的增量索引。
- `data/library/`：500 张演示素材，克隆后即可浏览。
- `data/index/incremental.npz`：与演示素材对应的索引，可直接用于重复分组。
- `data/photo_manifest.json`：图片来源、变换类型和同源分组记录。
- `data/test.jpg` 和 `data/test_other.jpg` 不随仓库提供；搜索时在网页选择查询图片，或用 `--query` 指定图库里的文件。

组员在项目目录运行以下命令（Windows PowerShell 或 cmd 均可）：

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe web_app.py
```

浏览器打开 `http://127.0.0.1:8765`。首次以图搜图或新增图片入库时仍需下载模型权重；模型、虚拟环境、备份和回收区不包含在仓库中。已克隆的组员在保存本地修改后运行 `git pull --ff-only origin main`，然后重启网页服务。

只共享这次确认的 500 张基准图片和索引；`.gitignore` 仍忽略其他本地数据，新导入图片不会自动提交。已跟踪的基准图片若被清理或替换，Git 会显示对应变化，提交前请检查。图片来源与使用说明见 [PHOTO_LIBRARY.md](PHOTO_LIBRARY.md)。

可运行 `python make_dataset.py` 程序化生成带同组标注的合成素材（默认 1500 张原图约 3300 张素材，含完全重复组、近重复变体组与独立单张），用于库容量演示与重复检测的准确率/召回率评测；同组真值写入 data/synthetic_labels.json，`--bases`、`--seed` 可控制规模与复现。

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

`indexed_search.py` 是基础版本；网页版和 `incremental_search.py` 使用可增量维护的
NPZ权威数据，并生成FAISS检索侧车。FAISS不可用时自动回退NumPy，不影响数据读取。

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

索引每处理25张新图写入一次检查点；更新中断后，重新执行会复用已保存向量。
SQLite保存稳定图片ID与入库状态，FAISS负责向量召回；文件哈希扫描仍会读取全部图片。
请勿同时运行多个更新进程。

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
- near：不同文件的向量余弦相似度至少 0.94，且 64 位差分感知哈希（dHash）距离不超过 6。该默认方案是保守基线；500 张图库的回归评测与调参结果见 `LIBRARY_EVALUATION_RESULTS.md`，未参与调参的最终留出集验收见 `FINAL_HOLDOUT_RESULTS.md`。
- similar：向量达到上述阈值，但感知哈希不通过，仅列入相似图片对，不据此合并。
- 其他图片保持独立。这些标签是算法判定，不是人工标准答案。

先把完全重复文件绑定，再按稳定文件名顺序进行全连接约束分组。新成员必须与组内
所有成员都通过判定，防止 A 像 B、B 像 C 就把 A/B/C 串联。代表图选择组内CLIP
平均相似度最高的medoid；推荐保留项综合分辨率、清晰度和曝光评分，并保留人工确认。

调整实验参数示例（重新生成报告）：

```bat
python group_duplicates.py --similarity 0.94 --hash-distance 6
```

分组优先使用FAISS阈值范围搜索，只保留达到CLIP阈值的候选边，不创建N×N邻接矩阵；FAISS不可用时自动使用NumPy分块扫描。每组最多输出200对证据并记录总数。dHash 对裁剪、边框等变化仍可能漏判；不能只为某张图片入组而放宽阈值，应在有来源记录的正负样本上评测。网页版提供综合质量评分、分组查看、人工选择保留项、软删除和恢复。

## 批量评测数据与阈值实验

```bat
python evaluate_duplicates.py prepare
python evaluate_duplicates.py evaluate
```

prepare 默认从 `data/library` 中选择 10 张 `photo_*` 图片（随机种子 42），按原图划分为 6 张调参原图和 4 张测试原图。每张生成 7 个版本：标准化原图、文件副本、50% 缩放、亮度 70%、四边各裁去 10%、右下角水印、JPEG 质量 55。原图预处理为 RGB，最长边不超过 1024；全部文件写入独立的 `data/evaluation`，原素材不变。已存在的评测目录不会覆盖，可用 `--dataset data/evaluation_v2` 指定新目录。

自备原图目录示例：

```bat
python evaluate_duplicates.py prepare --source data/originals --pattern "*" --count 10 --dataset data/evaluation_v2
python evaluate_duplicates.py evaluate --dataset data/evaluation_v2
```

`manifest.json` 记录来源文件、哈希、变换参数和分区；需检查被选原图确为独立素材，程序只能排除文件完全重复，无法自动保证不同原图之间不是近重复。所有同源版本是正样本对，不同来源是负样本对。调参集与测试集按来源分离，不进行跨分区比较。

evaluate 建立独立增量索引，比较 9 组候选阈值（余弦 0.90/0.94/0.97 × 哈希距离 6/12/18），仅根据调参集选择候选，再与默认阈值在测试集比较。优先选择调参集查准率至少 98% 的最高召回率方案；不达标时只报告 F1 最高候选。结果写入 `evaluation_report.md` 与 `evaluation_results.json`，含 TP/FP/FN/TN、查准率、召回率、F1、误判率、错误图片对，以及原图对各变换版本的检出情况。不会自动修改分组脚本阈值。

上述命令用于生成新的小型独立数据。仓库现有500张带来源标注的素材可直接复用既有向量，运行：

```bat
python evaluate_library.py
```

脚本严格核对清单、磁盘和索引哈希，仅用调参集选择阈值，再一次性报告回归分区；同时对比简单传递闭包与系统全连接约束聚类。当前回归结果：默认 0.94/6 与调参选出 0.96/4 均为 P/R/F1 100%（零误判零漏判）；全连接约束聚类 0 错误合并组、0 拆组，详见 `LIBRARY_EVALUATION_RESULTS.md` 和 JSON 原始结果。未参与调参的最终留出集验收见 `FINAL_HOLDOUT_RESULTS.md`。若依据测试结果继续改算法，需另留新的来源图用于最终验收。

```bat
python -m unittest test_evaluate_library test_evaluate_duplicates test_group_duplicates test_incremental_search -v
```

消融实验与真值交叉验证（量化四路信号的贡献，并核验程序化真值符合人工直觉）：

```bat
python ablation_study.py
python cross_validate_labels.py --make-template
python cross_validate_labels.py data/evaluation/annotation_template.csv
```

结果见 `ABLATION_RESULTS.md`；标注模板行序已乱序固定，可人工补标剩余行后重跑验证。

## 本地网页版

```bat
python web_app.py
```

启动后用浏览器打开提示的地址（默认 `http://127.0.0.1:8765`，可用 `--port` 修改）。服务仅监听本机，页面令牌每次启动随机生成。

- 素材库浏览：缩略图网格，支持文件名筛选、排序、按重复组合并显示。
- 导入图片：支持多选（JPG、PNG、WebP、BMP，单张上限 20 MB），同名图片自动另存，导入完成后自动更新索引。
- 以图搜图：选择或拖入查询图片即可搜索，查询图不会加入素材库；支持相似度下限过滤，并可两套阈值并排对比（同一查询图只编码一次，阈值过滤在本地完成）。
- 重复组查看与两套阈值对比：可修改相似度与哈希距离重新分析，两套条件并排对比；支持按组清理其余版本（可手动选择保留项，其余版本移入回收区并同步索引，可恢复，操作前有确认提示）。
- 素材清单导出：导出CSV或JSON，包含稳定ID、尺寸、格式、SHA-256、模型版本、重复组与质量评分。

网页端的索引更新采用单任务队列，任务运行期间不能进行其他导入或更新操作；关闭终端即停止服务。模型在首次搜索或处理新图时才加载，首次加载需联网下载权重。网页端调用重复分组时不写入 `duplicate_groups.json`，命令行版本的报告不受影响。

## 网页版测试用例

课程要求的 15 条软件测试用例覆盖网页版的输入校验、安全防护与端到端流程，用例说明见 [TEST_CASES.md](TEST_CASES.md)。

全套 57 项自动化用例一键全跑（含空库/单图/全重复分组边界，约 10 秒，不下载模型、不碰真实素材）：

```bat
python -m unittest discover -v
```

## 检查与验证（2026-10-21）

运行 `python -m unittest discover -v`：57 项全绿——网页/API 28 项、重复分组 8 项（含空库/单图/全重复边界）、增量索引 4 项、后端服务 11 项、评测与数据生成 6 项。测试全程使用临时图片与 FakeEncoder 替身，不下载模型；真实模型精度由 500 张库回归 + 留出集验收 + 消融实验三套真实数据实验独立背书，程序真值经人工肉眼交叉验证 100% 一致。

素材和分组分批显示；搜索命中总数与展示前 20 张分别统计。清理允许选择保留版本，使用签名分组快照，素材变化后拒绝执行。图片移入 `data/index/trash`，可在素材库恢复最近批次；回收区仍占用磁盘。两套阈值对比仅供查看。

合成条纹图片的单像素尺寸问题已修复，已有数据不会自动覆盖。合成素材用于容量测试，真实照片精度需要独立评测。详细记录见 [REVIEW_REPORT.md](REVIEW_REPORT.md)。
