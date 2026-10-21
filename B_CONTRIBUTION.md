# B 后端与模型部署个人贡献（77-gm）

> 详细的文件级归属、验收数据与答辩口径见 `B_PERSONAL_DELIVERABLES.md`。
> 个人贡献与实际代码、接口和测试的逐项对应见 `B_CODE_EVIDENCE.md`。
> 可运行 `powershell -ExecutionPolicy Bypass -File .\show_my_backend_work.ps1`
> 自动输出个人提交、新增文件和代码改动统计。

## 个人职责与完整交付范围

当前课程交付版本中，我负责B后端与模型部署，个人交付范围覆盖模型加载、图片向量化、
索引维护、以图搜图、重复归并、批量任务、质量推荐、阈值评测、后端接口、数据导出、
自动化测试和部署文档。主要完成内容如下：

1. 统一Python 3.12环境，将虚拟环境、pip缓存和模型权重放在D盘项目目录。
2. 修复Python 3.8兼容、Pillow API、Windows中文路径和新版PyTorch加载问题。
3. 增加CPU/GPU自动选择、离线权重复用和健康检查接口。
4. 增加持久FAISS稳定ID增删同步、SQLite稳定ID目录、分页查询和JSON/CSV审计导出。
5. 增加每25张检查点、任务进度、失败记录和中断后恢复。
6. 增加多指标质量评分、medoid代表图和逐对判定理由。
7. 增加独立测试、实际全连接聚类指标以及500/2000/5000规模的性能基准。
8. 使用FAISS阈值范围搜索生成去重候选边，移除N×N布尔邻接矩阵。
9. 增加OpenCLIP以文搜图、严格/校准推荐/宽松阈值预设和逐条命中解释。
10. 增加多视图dHash/pHash与RGB色度安全门，提升裁剪、水印鲁棒性并抑制误合并。

OpenCLIP作为第三方预训练模型提供基础视觉编码能力；模型部署、向量索引、任务恢复、
聚类、质量评分、评测、接口和部署逻辑均纳入个人后端交付与验收范围。

## 功能完成情况

| 课程模块 | 后端完成情况 | 验收证据 |
|---|---|---|
| 1 图像向量化与索引 | 已完成 | 批量OpenCLIP + NPZ + 持久FAISS稳定ID + SQLite |
| 2 近重复判定与归并 | 已完成 | 五信号判定、FAISS候选召回、完整链接约束、medoid、判定理由 |
| 5 批量导入与增量更新 | 已完成 | SHA-256复用、检查点、进度、重试与原子保存 |
| 6 阈值调节与对比 | 已完成 | 网页双阈值、独立校准/测试、聚类级指标 |
| 7 重复组管理 | 已完成 | 综合质量推荐、HMAC确认、软删除与恢复 |
| 10 浏览与导出 | 已完成 | 网格、SQLite分页、CSV/JSON审计字段 |

## 个人Git提交

```text
bfd8370 perf(backend): add cached search and measured pipeline benchmarks
66d8847 feat(dedup): add five-signal validation and final holdout
f09794e feat(retrieval): add text search explanations presets and crop robustness
5f3361b feat(index): add batch encoding and persistent stable-id FAISS
f5d3dc4 docs(backend): map every contribution to code evidence
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
4cf233f docs(backend): record catalog migration verification
496b0d1 perf(dedup): use FAISS range search for candidate pairs
41ff184 docs(backend): record calibrated metrics and verification
e482d51 feat(api): expose dedup candidate retrieval evidence
04e0a8a docs(backend): document deployment APIs and personal contribution
0f4ddd2 docs(backend): add auditable personal deliverables
98e151a docs(backend): define complete backend delivery scope
65e0d5c refactor(backend): own model and recycle services
```

查看可核验记录：

```powershell
git log --author="77-gm" --oneline --stat
```

## 已验证结果

- 真实模型：OpenCLIP `ViT-B-32-quickgelu` / `openai`，CPU输出512维单位向量。
- 真实检索：500张素材全部参与，持久FAISS侧车包含500个稳定ID。
- 最终留出评测：168张新生成变体；图片对Precision 99.77%、Recall 87.30%、F1 0.9312，见 `FINAL_HOLDOUT_RESULTS.md`。
- 自动化测试：使用Python 3.12虚拟环境运行 `python -m unittest discover -v`，当前54项测试全部通过。
- 性能数据：`PIPELINE_BENCHMARK_RESULTS.md`和`MODEL_BATCH_BENCHMARK_RESULTS.md`分别记录管线与真实模型耗时。
