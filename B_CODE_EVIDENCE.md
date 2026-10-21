# B 后端个人贡献代码证据表（77-gm）

本表把个人贡献声明逐项对应到可运行代码、接口、测试和结果文件。验收时可以按表中
路径直接定位，不需要只依赖文字说明。

| 个人贡献 | 核心代码 | 接入位置 | 自动化或结果证据 |
|---|---|---|---|
| OpenCLIP部署、CPU/GPU选择、D盘权重缓存 | `backend_core/model_service.py` | `indexed_search.py` 的 `load_model`、`encode_image` | `test_backend_core.py`、真实500张检索 |
| 图片批量向量化、归一化与坏图隔离 | `backend_core/model_service.py::encode_image_paths` | `incremental_search.py::Encoder.encode_many` | `ModelServiceTests`、真实批处理基准 |
| 持久FAISS稳定ID、增删同步与缓存 | `vector_index.py::persist_vector_index/rank_persistent` | `incremental_search.py`、`web_app.py` | `test_backend_services.py` |
| OpenCLIP以文搜图 | `backend_core/model_service.py::encode_texts` | `/api/search-text` | `test_backend_core.py`、`test_web_app.py` |
| FAISS去重候选范围召回 | `vector_index.py::threshold_pairs` | `group_duplicates.py` | `test_group_duplicates.py`、2886个候选对结果 |
| SQLite稳定ID、状态、分页过滤 | `metadata_store.py::MetadataStore` | 增量入库、目录和导出接口 | `test_backend_services.py`、`test_web_app.py` |
| 每25张检查点和中断恢复 | `incremental_search.py::update_index` | `/api/update`后台任务 | `test_incremental_search.py` |
| 任务进度和失败状态 | `web_app.py::submit/update_progress` | `/api/jobs/<id>` | `test_web_app.py` |
| 五信号近重复判定和完整链接约束 | `group_duplicates.py` | `/api/groups`、`/api/compare` | `test_group_duplicates.py`、最终留出集 |
| 阈值预设和命中解释 | `backend_core/retrieval_policy.py` | `/api/presets`、搜索结果卡片 | `test_backend_core.py`、`test_web_app.py` |
| medoid代表图和质量评分 | `group_duplicates.py::inspect_image/generate_groups` | 重复组接口和页面 | `test_group_duplicates.py` |
| HMAC审查签名 | `backend_core/recycle_service.py::ReviewSigner` | `LibraryApp.sign_review` | `RecycleServiceTests`、`test_web_app.py` |
| 软删除、失败回滚和恢复 | `backend_core/recycle_service.py::RecycleBinService` | `/api/cleanup`、`/api/restore` | `RecycleServiceTests`、TC18-TC22 |
| 健康诊断 | `web_app.py::health` | `/api/health` | `test_web_app.py::test_tc23*` |
| CSV/JSON审计导出 | `web_app.py::export/export_json` | `/api/export` | `test_web_app.py::test_tc24*` |
| 独立校准/测试和聚类指标 | `evaluate_library.py` | 命令行评测 | `test_evaluate_library.py`、`LIBRARY_EVALUATION_RESULTS.*` |
| 500/2000/5000管线性能基准 | `benchmark_pipeline.py` | 命令行基准 | `PIPELINE_BENCHMARK_RESULTS.*` |
| 真实OpenCLIP批处理基准 | `benchmark_model_batch.py` | 命令行基准 | `MODEL_BATCH_BENCHMARK_RESULTS.*` |
| 算法冻结后最终留出验收 | `final_holdout_validation.py` | 固定种子可重建 | `FINAL_HOLDOUT_RESULTS.*` |

## 一键核验

```powershell
Set-Location "D:\master-1 paper\image-library-backend"
.\.venv\Scripts\python.exe -m unittest discover -v
powershell -ExecutionPolicy Bypass -File .\show_my_backend_work.ps1
```

## 第三方组件边界

- OpenCLIP提供预训练模型结构和权重。
- FAISS提供底层近邻索引实现。
- Pillow负责图片解码，SQLite负责元数据持久化。
- 模型部署、数据流程、索引维护、聚类规则、质量评分、恢复事务、评测口径、接口和测试
  均由本项目后端代码组织实现。

