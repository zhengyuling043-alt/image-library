# 真实照片图库（2026-10-07）

## 当前规模：500 张

### 小组共享

仓库随代码共享 `data/library/` 的 500 张基准图片、`data/photo_manifest.json` 来源清单和 `data/index/incremental.npz` 索引，合计约 139 MiB。克隆后启动 `python web_app.py` 即可浏览图库和运行重复分组。首次搜索仍需下载模型权重。

这些照片来自 Pexels，同源变体用于课程项目的重复检测和演示。逐图来源见清单，照片不属于项目原创，也不随项目代码重新授权；适用 [Pexels License](https://www.pexels.com/license/)。不要把这些图片重新发布为素材或壁纸下载平台。

备份、临时下载、回收区、个人新增图片和模型缓存不随本次提交共享。若组员原有未跟踪的同名数据，Git 拉取可能提示覆盖冲突，应先备份移开同名文件再拉取。

第三批从公开 Pexels 照片中采用 31 张新原图，主题包含瀑布、森林、鸟类和沙漠；加上对应测试变体共新增 127 张。现在合计 108 个原图来源、500 张图片。原有 373 张文件保持不变，新编号从 `photo_0078` 到 `photo_0108`。

为精确达到 500 张，最后一组只生成所需数量的版本，实际成员均记录在 `data/photo_manifest.json`。本轮候选下载 44 张，其余候选仅保存在暂存下载目录，不计入当前图库。备份位于 `data/backups/before_photo_expansion_500_20261007/`。

## 第二批扩充

新增下载 47 张真实原图，包含海边、山湖、花卉、城市建筑、食物等；48 个候选中 1 个未下载成功。原图及部分来源的亮度、缩放、裁剪、压缩版本共新增 199 张。图库扩充到 77 个原图来源、373 张图片，现有文件名和内容保持不变，新编号从 `photo_0031` 继续。

下载记录：`data/photo_expansion_20261007/downloads.json`；完整来源与分组记录仍为 `data/photo_manifest.json`。扩充前 174 张图库及索引、清单备份在 `data/backups/before_photo_expansion_20261007/`。

以下是第一批的准备记录。

本轮采用 21 张原有 Pexels 照片，加上 9 张本次下载照片，共 30 个原图来源、174 张图库图片。1 个下载链接失败，已记录；未把错误页面当作图片入库。Pexels 搜索页要求浏览器验证，所以本轮通过公开图片链接下载，并非完整抓取指定搜索页。

## 命名

- `photo_0001_original.jpg`：原图的标准化版本，最长边不超过 1600 像素。
- 同编号的 `copy`、`brightness`、`resize`、`crop`、`jpeg`、`watermark` 分别表示完全复制、亮度、缩放、裁剪、压缩、水印版本。
- 每第五个来源只保留原图，作为独立素材。
- `data/photo_manifest.json` 保存每个文件的来源网址、原文件名或 Pexels 编号、来源哈希、变换类型及来源级调参/测试划分。

这些分组表示同源变换关系，不等同于模型预测；不同来源之间可能存在近重复，评测前仍需人工核对。本批用于真实图片流程和演示，30 张原图不代表已满足课程的大规模验收要求。

## 备份与使用

旧图库及其索引完整保存在 `data/backups/before_photo_refresh_20261007/`；旧合成标注和旧评测目录仍保留，不能当成新图库的评测结果。

重新启动 `python web_app.py` 后使用新图库。增量索引已与新文件名同步。

如果要为新图库重新生成独立评测，请使用新的输出目录：

```bat
python evaluate_duplicates.py prepare --pattern "photo_*_original.jpg" --count 30 --dataset data/evaluation_photos_20261007
python evaluate_duplicates.py evaluate --dataset data/evaluation_photos_20261007
```

图片来源：Pexels，逐张链接见本地清单。库中派生图片仅用于本项目重复检测测试。
