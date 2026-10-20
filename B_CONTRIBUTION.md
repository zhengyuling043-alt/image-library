# B 后端与模型部署个人贡献（77-gm）

## 项目原有能力与个人贡献边界

项目原有能力包括OpenCLIP图片编码、NPZ索引、基础增量更新、以图搜图、
SHA-256/dHash/余弦相似度分组、阈值对比、软删除恢复和网页界面。

本分支的个人贡献是把上述原型升级为可部署、可恢复、可测量的后端：

1. 统一Python 3.12环境，将虚拟环境、pip缓存和模型权重放在D盘项目目录。
2. 修复Python 3.8兼容、Pillow API、Windows中文路径和新版PyTorch加载问题。
3. 增加CPU/GPU自动选择、离线权重复用和健康检查接口。
4. 增加FAISS检索、SQLite稳定ID目录、分页查询和JSON/CSV审计导出。
5. 增加每25张检查点、任务进度、失败记录和中断后恢复。
6. 增加多指标质量评分、medoid代表图和逐对判定理由。
7. 增加独立测试、实际全连接聚类指标以及500/2000/5000规模的性能基准。
8. 使用FAISS阈值范围搜索生成去重候选边，移除N×N布尔邻接矩阵。

OpenCLIP模型已有功能仅是将图片编码成512维向量；上述索引、恢复、聚类、
评分、评测、接口和部署逻辑均属于本分支的工程实现。

## 功能完成情况

| 课程模块 | 后端完成情况 | 验收证据 |
|---|---|---|
| 1 图像向量化与索引 | 已完成 | OpenCLIP + NPZ + FAISS + SQLite |
| 2 近重复判定与归并 | 已完成 | 三档判定、FAISS候选召回、全连接约束、medoid、判定理由 |
| 5 批量导入与增量更新 | 已完成 | SHA-256复用、检查点、进度、重试与原子保存 |
| 6 阈值调节与对比 | 已完成 | 网页双阈值、独立校准/测试、聚类级指标 |
| 7 重复组管理 | 已完成 | 综合质量推荐、HMAC确认、软删除与恢复 |
| 10 浏览与导出 | 已完成 | 网格、SQLite分页、CSV/JSON审计字段 |

## 个人Git提交

```text
68b9235 fix(runtime): support Python 3.8+ and stable model cache
68075f2 feat(api): add reproducible deployment and health diagnostics
066e54d feat(index): add FAISS SQLite and resumable batch updates
2eee8d4 fix(storage): support Unicode paths and close SQLite handles
643ca4e feat(dedup): explain matches and score keeper quality
48aec83 fix(model): load trusted OpenAI TorchScript weights on modern PyTorch
25ec934 feat(evaluation): add cluster metrics and scale benchmarks
7884818 feat(export): add paginated catalog and auditable exports
b1cf702 chore(storage): ignore generated index artifacts
33e20dc test(api): retry transient Windows loopback reset
b6d7b0d fix(catalog): migrate existing NPZ metadata on startup
dbfd6e1 feat(evaluation): validate dedup on 500-image holdout
496b0d1 perf(dedup): use FAISS range search for candidate pairs
```

查看可核验记录：

```powershell
git log --author="77-gm" --oneline --stat
```

## 已验证结果

- 真实模型：OpenCLIP `ViT-B-32-quickgelu` / `openai`，CPU输出512维单位向量。
- 真实检索：500张素材全部参与，FAISS返回同源亮度、副本、原图、缩放、水印和JPEG版本为前列结果。
- 真实去重评测：500张素材按来源隔离调参集与测试集；全连接聚类查准率98.55%、召回率92.39%、F1为0.9537，见 `LIBRARY_EVALUATION_RESULTS.md`。
- 自动化测试：使用Python 3.12虚拟环境运行 `python -m unittest discover -v`，当前43项测试全部通过。
- 性能数据：见 `BENCHMARK_RESULTS.md`，明确区分向量检索耗时与图片编码耗时。
