# Phase 3: Feature Audit Cache

## 涉及文件

- `features/audit.py`

## 问题

`run_all_audits()` 执行 streaming-vs-batch 回归测试，在全量数据上对 ~15 个采样日期逐日对比。
所有结果在内存中累积，中断后全部丢失。

虽然 Phase 3 运行时间相对较短（< 2 分钟），但它依赖于 Phase 2 的输出（`results/features_{tier}.parquet`）。
当 Phase 2 被重新运行后，Phase 3 一定需要重新运行；但如果 Phase 2 没有变化，Phase 3 也不应
重新运行。

当前 `run_all.py` 总是重新调用 Phase 3，即使输入没有变化。

## 方案

### 输出文件保持不变

```
results/audit/
├── feature_parity_summary.json
├── validation_summary.json
└── lookahead_checklist_onepager.md
```

### 添加 cache manifest

在 `run_all_audits()` 完成后，写出 manifest：

```python
from data.cache_utils import build_cache_manifest, write_cache_manifest

def run_all_audits(...) -> dict:
    # ... existing computation ...

    # 写出 cache manifest
    features_path = RESULTS_DIR / f"features_{tier}.parquet"
    manifest = build_cache_manifest(
        phase=f"3_{tier}",
        parameters={"tier": tier, "small_only": small_only, "full_dates": full_dates},
        input_paths=[
            SIGNALS_PARQUET,
            PRICE_CACHE_DIR / "_manifest.json",
            features_path,                       # Phase 2 输出
        ],
        source_funcs=[run_all_audits, run_streaming_vs_batch_test],
    )
    write_cache_manifest(
        manifest,
        CACHE_MANIFEST_DIR / f"3_{tier}.json",
    )

    return results
```

### `run_all.py` 的 `_skip()` 处理

```python
# Phase 3 的 artifact 检查
# ARTIFACTS["3_enhanced"] = AUDIT_DIR / "feature_parity_summary.json"
# 配合 manifest：CACHE_MANIFEST_DIR / "3_enhanced.json"
```

## 缓存失效条件

- `data/cache/signals.parquet` 内容变化
- `data/cache/prices/_manifest.json` 内容变化
- `results/features_{tier}.parquet` 内容变化（Phase 2 重跑后自动失效）
- `tier`、`small_only`、`full_dates` 参数变化
- audit 函数源码变化

## 预期节省

- 每次跳过 Phase 3：~1-2 分钟
