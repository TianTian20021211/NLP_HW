# Phase 5a: Single-Feature IC Cache

## 涉及文件

- `backtest/single_feature_ic.py`

## 问题

`run_single_feature_ic()` 在内存中循环计算 14 features × 5 horizons × 5 signal types = 350 个
IC 组合，所有结果累积在 `list` 中，最终一次性写成 3 个 parquet 文件：

```
results/ic/
├── ic_summary_{universe}.parquet
├── ic_yearly_{universe}.parquet
└── ic_sector_split_{universe}.parquet
```

如果 pipeline 在 combo 300/350 时中断，前面 299 个 combo 的计算全部丢失。

## 方案

### 整体缓存（粗粒度，优先实施）

检查 3 个输出文件是否都存在 + cache manifest 是否有效，命中则跳过整个 Phase 5a。

在 `run_single_feature_ic()` 完成后写出 manifest：

```python
from data.cache_utils import build_cache_manifest, write_cache_manifest

def run_single_feature_ic(features_path, universe_name, ...):
    # ... existing computation ...
    # 写出 cache manifest
    manifest = build_cache_manifest(
        phase=f"5a_{universe_name}",
        parameters={
            "universe_name": universe_name,
            "features": list(short_list_features),
            "horizons": list(HORIZONS),
            "signal_types": list(SIGNAL_TYPES),
        },
        input_paths=[
            features_path,
            PRICE_CACHE_DIR / "_manifest.json",
        ],
        source_funcs=[run_single_feature_ic, _compute_ic],
    )
    write_cache_manifest(manifest, CACHE_MANIFEST_DIR / f"5a_{universe_name}.json")
    return results
```

### Per-combo 增量 checkpoint（细粒度，可选后续优化）

如果需要更细粒度的断点续跑，可以每 10 个 combo 写一次中间文件：

```python
# 在 combo 循环中
if combo_idx % 10 == 0:
    partial_df.to_parquet(output_dir / f".ic_partial_{universe_name}.parquet")
```

然后在函数开头检查 partial 文件，加载已完成的 combo 结果，只计算剩余的。

**先实施粗粒度缓存**。如果 350 combo × multi-universe 的运行时仍然过长，再加 per-combo checkpoint。

## 缓存失效条件

- `results/features_{tier}.parquet` 内容变化
- `data/cache/prices/_manifest.json` 内容变化
- `universe_name` 参数变化
- IC 计算函数源码变化

## 预期节省

- 每次跳过 Phase 5a per universe：节省 350 combo 的 IC 计算
- 3 universes × 3 files = 9 个缓存输出文件
