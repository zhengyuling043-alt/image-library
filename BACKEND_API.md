# 后端接口说明

服务仅监听 `127.0.0.1`。除只读GET接口外，POST请求需要主页注入的
`X-App-Token`，用于防止跨站请求。

| 方法 | 路径 | 功能 |
|---|---|---|
| GET | `/api/health` | Python、模型、设备、FAISS和索引状态 |
| GET | `/api/library` | 素材库状态与缩略图元数据 |
| GET | `/api/catalog?limit=50&offset=0&q=` | SQLite分页目录与文件名过滤 |
| GET | `/api/jobs/<id>` | 后台任务状态及入库百分比 |
| POST | `/api/import` | Base64图片导入，单张上限20MB |
| POST | `/api/update` | 增量更新；可传 `{"rebuild": true}` 全量重建 |
| POST | `/api/search` | 查询图片编码后使用FAISS/NumPy召回 |
| POST | `/api/groups` | 重复和近重复分组，返回逐对判定理由 |
| POST | `/api/compare` | 两套阈值并排比较 |
| POST | `/api/cleanup` | 带HMAC审查快照的软删除 |
| POST | `/api/restore` | 恢复一个回收批次 |
| POST | `/api/export` | CSV；传 `format=json` 输出JSON |

搜索响应中的 `backend` 会明确返回 `faiss-index-flat-ip` 或
`numpy-dot-product`。分组响应包含 `representative`、`quality_scores`、
`suggested_keep`、`keep_reason` 和每个图片对的 `reason`，用于解释系统判断。
分组顶层同时返回 `candidate_backend`、`candidate_pair_count` 与
`similar_pair_count`，页面可直接展示FAISS候选召回方式和各阶段数量。
