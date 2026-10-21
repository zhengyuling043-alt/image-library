# 最终留出集评测

算法与阈值冻结后生成的固定种子合成留出集；未用于调参，不能替代真实用户数据外部验证。

- 固定随机种子：`20261024`；独立原图：24；变体后共 168 张。
- 冻结阈值：CLIP相似度 ≥ 0.86，多视图dHash距离 ≤ 8。
- 图片对：Precision 99.77%，Recall 87.30%，F1 0.9312。
- 完整组：Precision 99.75%，Recall 78.97%，F1 0.8815，纯度 99.40%。

| 变换 | 检出/应检出 | 召回率 |
|---|---:|---:|
| copy | 24 / 24 | 100.00% |
| resize | 23 / 24 | 95.83% |
| brightness | 21 / 24 | 87.50% |
| crop | 23 / 24 | 95.83% |
| watermark | 23 / 24 | 95.83% |
| jpeg | 23 / 24 | 95.83% |

详细错误对见 `FINAL_HOLDOUT_RESULTS.json`。评测数据位于忽略提交的 `.runtime/final-holdout`，可由脚本固定种子重建。
命令：`.venv\Scripts\python.exe final_holdout_validation.py`。
