# B后端答辩演示顺序

1. 运行 `./start_backend.ps1`，访问 `/api/health`，展示Python、模型、设备、512维和持久FAISS状态。
2. 在素材库展示500张素材；导入新图并更新，展示后台进度、ready/pending/failed状态。
3. 上传已有变体执行以图搜图，指出结果卡片的类别、CLIP分数、dHash/pHash/RGB判定依据。
4. 输入英文描述执行以文搜图，说明图文共享向量空间及中文权重限制。
5. 打开重复组，切换严格/校准推荐/宽松预设，展示双阈值对比。
6. 展开组内证据，说明完整链接如何阻止传递链；查看质量分与推荐保留理由。
7. 执行软删除后打开回收区恢复，说明HMAC审查签名和失败回滚。
8. 展示 `FINAL_HOLDOUT_RESULTS.md`、`PIPELINE_BENCHMARK_RESULTS.md` 和54项测试。
9. 运行 `git log --author="77-gm" --oneline`，展示个人提交记录。

答辩时明确区分：OpenCLIP提供预训练编码能力；后端部署、索引、增量、聚类、解释、恢复、评测和API为个人交付。
