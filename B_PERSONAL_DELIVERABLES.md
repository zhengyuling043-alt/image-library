# B 后端个人交付清单（77-gm）

本文件用于课程答辩、个人工作量说明和 Pull Request 审查。B后端负责人为
`77-gm`，个人开发分支为 `backend/b-model-index`。当前交付版本的模型部署、索引、
检索、去重、任务恢复、评测、接口、测试和部署均纳入个人后端交付范围。

## 一、个人新增的核心后端文件

以下文件由个人开发分支新增，并有独立Git提交记录：

| 文件 | 个人实现内容 |
|---|---|
| `vector_index.py` | FAISS/NumPy双后端向量检索、阈值范围候选召回、索引序列化 |
| `metadata_store.py` | SQLite图片目录、稳定ID、状态和逻辑删除、分页筛选 |
| `runtime_compat.py` | Python/Pillow运行时兼容处理 |
| `evaluate_library.py` | 500张真实素材的校准集/测试集隔离评测和聚类级指标 |
| `benchmark_backend.py` | 500/2000/5000规模向量检索耗时和内存基准 |
| `test_backend_services.py` | FAISS、SQLite和后端服务测试 |
| `test_evaluate_library.py` | 独立评测流程、数据完整性和指标测试 |
| `setup_backend.ps1` | D盘Python虚拟环境和依赖部署脚本 |
| `start_backend.ps1` | 后端一键启动脚本 |
| `BACKEND_API.md` | 后端接口契约 |
| `BACKEND_DEPLOYMENT.md` | 模型、依赖和启动部署说明 |
| `BENCHMARK_RESULTS.md/json` | 可复核性能结果和原始数据 |
| `LIBRARY_EVALUATION_RESULTS.md/json` | 可复核准确率、召回率、F1和错误分析 |

## 二、个人负责的后端主流程文件

以下文件构成当前后端主流程，由B负责人完成工程化实现、集成和最终验收。具体代码
记录可用 `git diff origin/main..backend/b-model-index -- <文件>` 核验。

| 文件 | 个人负责内容 |
|---|---|
| `indexed_search.py` | OpenCLIP部署、D盘离线权重、CPU/GPU选择、新版PyTorch兼容、FAISS检索 |
| `incremental_search.py` | 增量入库、检查点恢复、任务状态、FAISS与SQLite同步 |
| `group_duplicates.py` | SHA-256+dHash+CLIP判定、FAISS范围召回、medoid、质量评分和解释证据 |
| `evaluate_duplicates.py` | 阈值实验、校准/测试隔离和聚类级评测接入 |
| `web_app.py` | 本地HTTP API、健康诊断、目录迁移、分页、审计导出和召回证据 |
| `test_incremental_search.py` | 中断恢复、SQLite和FAISS同步测试 |
| `test_group_duplicates.py` | 传递链、质量推荐和候选召回测试 |
| `test_web_app.py` | 健康、分页、迁移、导出和Windows兼容测试 |

## 三、课程六个模块中的个人后端成果

1. **模块1 图像向量化与索引**：部署OpenCLIP，新增FAISS检索、SQLite元数据目录和稳定ID。
2. **模块2 近重复判定与归并**：实现稀疏候选召回、全连接约束聚类、medoid和判定理由。
3. **模块5 批量导入与增量更新**：实现SHA-256复用、检查点、失败重试和中断恢复。
4. **模块6 阈值调节与效果对比**：实现独立校准/测试、图片对指标和聚类级指标。
5. **模块7 重复组管理**：实现清晰度、曝光和分辨率综合质量评分及保留推荐。
6. **模块10 浏览与导出**：实现SQLite分页过滤和带模型、哈希、分组、质量分的审计导出。

## 四、可量化验收结果

- OpenCLIP：`ViT-B-32-quickgelu` / `openai`，输出512维归一化向量。
- 自动化测试：43项通过。
- 真实素材独立测试：全连接聚类 Precision 98.55%、Recall 92.39%、F1 0.9537。
- 真实500张分组：FAISS范围搜索召回2886个候选对，避免构造N×N布尔邻接矩阵。
- 性能测试：已保存500、2000、5000向量规模的检索耗时和内存结果。

## 五、答辩时的准确表述

建议表述：

> 我负责B后端与模型部署，承担当前交付版本的后端设计、工程实现、系统集成、测试与
> 部署。完成内容包括OpenCLIP模型部署、FAISS+SQLite索引、可恢复批量入库、稀疏去重
> 候选召回、综合质量评分、聚类级评测、健康诊断、审计导出和自动化测试。

## 六、Git核验命令

```powershell
git log origin/main..backend/b-model-index --author="2026147135@qq.com" --oneline
git diff --stat origin/main..backend/b-model-index
git diff origin/main..backend/b-model-index -- vector_index.py metadata_store.py group_duplicates.py web_app.py
```

