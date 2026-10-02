# 本地网页版测试用例（15 条）

对象：`web_app.py` + `web/`（本地网页版素材库）。
自动化实现：`test_web_app.py`（单元测试框架 unittest，全部使用临时目录与替代编码器，不下载模型、不改动用户素材）。

运行方式：

```bat
python -m unittest test_web_app -v
```

15 条用例全部通过（2026-10-02，Python 3.14）。核心算法（增量索引、重复分组、阈值评测）另有 7 条单元测试：`python -m unittest test_evaluate_duplicates test_group_duplicates test_incremental_search`。

## 用例列表

| 编号 | 模块 | 用例名称 | 前置条件 | 操作步骤 | 预期结果 | 自动化方法 |
|---|---|---|---|---|---|---|
| TC01 | 路径安全 | 拒绝路径穿越与非法文件名 | 应用已启动 | 依次传入 `../x.jpg`、`a/b.jpg`、`a\b.jpg`、`..`、`.`、`:x`、空串、`None`、`a:b.jpg` | 全部抛出 ValueError；合法名 `ok.jpg` 返回素材目录内路径 | `test_tc01_safe_file_rejects_traversal` |
| TC02 | 图片解码 | 超过大小上限的负载被拒 | 无 | 传入长度超过 20 MB 对应 Base64 上限的字符串 | 抛出 ValueError，提示单张图片上限 | `test_tc02_decode_image_rejects_oversize` |
| TC03 | 图片解码 | 非法与损坏数据被拒 | 无 | 分别传入非法 Base64 字符串、损坏的图片字节 | 均抛出 ValueError | `test_tc03_decode_image_rejects_corrupt_data` |
| TC04 | 图片解码 | 有效图片正常通过 | 无 | 传入有效 PNG 的 Base64 | 返回与输入一致的原始字节 | `test_tc04_decode_image_accepts_valid_png` |
| TC05 | 上传校验 | 不支持的格式被拒 | 素材目录为空 | 依次上传 `.gif`、`.txt`、无扩展名文件 | 均抛出 ValueError，不落盘 | `test_tc05_upload_rejects_unsupported_format` |
| TC06 | 上传校验 | 文件名清洗与保留名处理 | 素材目录为空 | 上传 `co<n>:?.jpg`；再上传 `CON.jpg` | 前者存为 `co_n___.jpg`（非法字符替换为 `_`）；后者存为 `image_CON.jpg`（避开 Windows 保留名） | `test_tc06_upload_sanitizes_filename` |
| TC07 | 上传校验 | 同名图片自动另存 | `same.png` 已存在 | 再次上传同名图片 | 新文件命名为 `same_2.png`，原文件不被覆盖 | `test_tc07_upload_autorenames_duplicates` |
| TC08 | 任务队列 | 任务互斥 | 一个任务正在运行 | 运行期间再提交新任务 | 抛出 ValueError，提示已有任务运行 | `test_tc08_tasks_are_serialized` |
| TC09 | 参数校验 | 分组阈值越界被拒 | 无 | 相似度传 `-0.1`、`1.5`、`NaN`、`inf`；哈希距离传 `-1`、`65` | 均抛出 ValueError，不执行分组 | `test_tc09_groups_rejects_invalid_thresholds` |
| TC10 | 搜索 | 空索引给出可读提示 | 索引不存在 | 用任意图片发起搜索 | 抛出 ValueError，提示先导入并更新索引 | `test_tc10_query_requires_index` |
| TC11 | 素材快照 | ready / pending / failed 三态 | 索引中 `good.png` 为最新哈希、`changed.jpg` 为过期哈希；磁盘上另有损坏文件 `broken.png` | 调用快照接口 | 三个文件分别判定为 ready、pending、failed | `test_tc11_snapshot_reports_three_states` |
| TC12 | 清单导出 | CSV 内容正确且防公式注入 | 库中有 `ok.jpg` 与 `=evil.jpg` | 导出这两张（重复传一次）；再导出空列表、不存在的文件 | 表头正确；`=evil.jpg` 被加 `'` 前缀防止电子表格执行；重复项去重；空列表与未知文件均报 ValueError | `test_tc12_export_csv_and_formula_guard` |
| TC13 | HTTP 安全 | 令牌鉴权与 Host 校验 | 服务已启动 | 1) GET 首页；2) 无令牌 POST `/api/update`；3) 错误令牌 POST；4) 伪造 Host 的 POST | 首页 200 且 `__TOKEN__` 被替换为真实令牌；后三者均返回 403 | `test_tc13_home_token_and_authorization` |
| TC14 | 端到端 | 导入 → 更新 → 搜索全链路 | 素材目录为空 | HTTP 导入 `sample.png`；POST 更新索引并轮询任务；GET 素材库；POST 以图搜图 | 导入返回 pending；更新任务完成且新增 1；快照 total=1、ready=1；搜索第一名就是 `sample.png`，得分 1.0 | `test_tc14_import_update_search_end_to_end` |
| TC15 | HTTP 接口 | 导出、缩略图与错误处理 | 库中已有 `sample.png` | POST 导出；GET 缩略图；GET 未知路径；缩略图传穿越路径；POST 非对象 JSON | 导出 200 且带 UTF-8 BOM；缩略图 200 且为 JPEG；未知路径 404；穿越路径 400；格式错误 400 | `test_tc15_export_thumbnail_and_error_pages` |

## 扩展用例（新增功能）

核心功能迭代时补充的用例，不属于课程要求的 15 条，随代码一并维护。

| 编号 | 模块 | 用例名称 | 前置条件 | 操作步骤 | 预期结果 | 自动化方法 |
|---|---|---|---|---|---|---|
| TC16 | 重复组清理 | 批量清理组内多余版本 | 库中有近重复组（master.png 与其缩小版）及独立素材 | HTTP 导入 3 张图并更新索引；POST `/api/cleanup` 删除组内非推荐项；再对推荐项与独立素材发起清理 | 第一轮删除成功且索引同步（deleted=1）；重新分组后无候选组；第二轮全部被拒绝（非组内非推荐项不可删） | `test_tc16_cleanup_removes_extra_versions` |
| TC17 | 搜索阈值对比 | 搜索返回全量排序得分 | 索引含 25 张图，向量与查询可精确计算 | 构造索引后直接调用查询 | 返回全部 25 条结果（不止前 20），按得分降序、同分按文件名排序——前端据此做单阈值过滤与双阈值并排对比，且只需一次模型编码 | `test_tc17_query_returns_all_sorted_results` |

## 用例设计说明

- **覆盖维度**：输入校验（TC02～TC07、TC09）、安全防护（TC01、TC13、TC15 的穿越与注入）、状态与数据正确性（TC10～TC12）、任务调度（TC08）、端到端业务流程（TC14、TC15）。
- **测试替代**：编码器使用固定单位向量的替身（FakeEncoder），使「导入→更新→搜索」在毫秒级完成且可精确断言得分；快照测试通过构造真实/过期/损坏三种文件验证状态机。
- **边界与异常**：每条功能用例都包含至少一个异常分支，验证报错信息可读且不产生副作用（不落盘、不覆盖原索引）。
