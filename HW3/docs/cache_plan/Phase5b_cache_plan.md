# Phase 5b: Quintile/Decile Analysis Cache

## 涉及文件

- `backtest/quintile.py`

## 问题

`run_quintile_analysis()` 与 Phase 5a 模式相同——在内存中循环计算大量 combo：

- **Decile**: 5 horizons × 5 signal types = 25 combos（仅 ATCClassifierScore）
- **Quintile**: 14 features × 5 horizons × 5 signal types = 350 combos

所有结果累积在内存中，最终一次性写出：

```
results/quintile/
├── decile_summary_{universe}.parquet
├── quintile_summary_{universe}.parquet
└── quintile_equity_curves_{universe}.parquet
```

中断后所有 combo 丢失。

## 方案

### 整体缓存（粗粒度）

检查 3 个输出文件 + 1 个 meta JSON 是否存在，配合 cache manifest 验证有效性。

```python
from data.cache_utils import build_cache_manifest, write_cache_manifest

def run_quintile_analysis(features_path, universe_name, ...):
    # ... existing computation ...

    manifest = build_cache_manifest(
        phase=f"5b_{universe_name}",
        parameters={
            "universe_name": universe_name,
            "features": list(quintile_features),
            "horizons": list(HORIZONS),
            "signal_types": list(SIGNAL_TYPES),
        },
        input_paths=[
            features_path,
            PRICE_CACHE_DIR / "_manifest.json",
        ],
        source_funcs=[run_quintile_analysis, _compute_decile, _compute_quintile],
    )
    write_cache_manifest(manifest, CACHE_MANIFEST_DIR / f"5b_{universe_name}.json")
    return results
```

### 输出文件

```
results/quintile/
├── decile_summary_{universe}.parquet      # 25 combos
├── quintile_summary_{universe}.parquet    # 350 combos
├── quintile_equity_curves_{universe}.parquet
└── quintile_meta_{universe}.json
```

## 缓存失效条件

- `results/features_{tier}.parquet` 内容变化
- `data/cache/prices/_manifest.json` 内容变化
- `universe_name` 参数变化
- quintile 计算函数源码变化

## 预期节省

- 每次跳过 Phase 5b per universe：节省 375 combo 的分组和 IC 计算
- 3 universes 全部命中时节省约 5-10 分钟
