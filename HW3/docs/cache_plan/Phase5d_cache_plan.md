# Phase 5d: Robustness Checks Cache

## 涉及文件

- `backtest/robustness.py`

## 问题

`run_all_robustness()` 执行约 11 个独立的 robustness 分析 section。每个 section 在完成后
写出自己的 parquet 文件（一共 ~12 个文件）。这是所有 Phase 中已有最好部分恢复能力的：

- 如果 process 在 "market cap buckets" section 被杀掉，"subperiod IC" 和 "sector neutral"
  的结果已经保存在磁盘上。

但仍有问题：

1. **Section 内无 checkpoint**：每个 section 内部仍循环 14 features × 5 horizons × 5 signal types，
   全部在内存中累积，section 内中断全部丢失。
2. **`run_all.py` 不跳过**：即使所有 12 个文件都存在且有效，Phase 5d 仍然被无条件重跑。
3. **跨 section 共享的数据无缓存**：如 `compute_momentum_features` 的结果只对 beta_window
   40 和 90 做了缓存（`results/cache/momentum_beta_{window}.parquet`），其他 section 重跑时
   仍需重新计算。

## 方案

### Per-section cache manifest

在 `run_all_robustness()` 中，每 section 运行前检查对应输出文件 + manifest，命中则跳过：

```python
from data.cache_utils import validate_cache_manifest, build_cache_manifest, write_cache_manifest

SECTIONS = [
    ("subperiod_ic", _run_subperiod_ic_section),
    ("subperiod_quintile", _run_subperiod_quintile_section),
    ("sector_neutral", _run_sector_neutral_section),
    ("mcap_buckets", _run_mcap_buckets_section),
    ("weighting", _run_weighting_section),
    ("ofat_quantile", _run_ofat_quantile_section),
    ("ofat_cost", _run_ofat_cost_section),
    ("ofat_lookback", _run_ofat_lookback_section),
    ("weekly_timing", _run_weekly_timing_section),
    ("label_purge_gap", _run_label_purge_gap_section),
    ("beta_window", _run_beta_window_section),
]

def run_all_robustness(features_path, universe_name, ...):
    for section_name, section_func in SECTIONS:
        manifest_path = CACHE_MANIFEST_DIR / f"5d_{section_name}_{universe_name}.json"
        output_path = output_dir / f"robustness_{section_name}.parquet"

        if output_path.exists() and validate_cache_manifest(manifest_path):
            print(f"  skip robustness/{section_name} (cached)")
            continue

        result = section_func(...)
        result.to_parquet(output_path)

        manifest = build_cache_manifest(
            phase=f"5d_{section_name}_{universe_name}",
            parameters={"universe_name": universe_name, "section": section_name},
            input_paths=[features_path, PRICE_CACHE_DIR / "_manifest.json"],
            source_funcs=[section_func],
        )
        write_cache_manifest(manifest, manifest_path)
```

### Section 内不需要额外 checkpoint

由于每个 section 输出是独立的 parquet 文件，且 cache manifest 机制确保 section 级别的
跳过和恢复，不需要在 section 内部添加更细粒度的 checkpoint。

如果某个 section（如 `subperiod_ic`）单独运行时间过长，可以后续将该 section 拆分为更小的
子单元。

### bootstrap CI 特殊处理

`robustness_bootstrap_ci.json` 是跨 section 的汇总结果。单独为其写 manifest：

```python
manifest = build_cache_manifest(
    phase=f"5d_bootstrap_{universe_name}",
    parameters={"universe_name": universe_name},
    input_paths=[features_path],
    source_funcs=[_run_bootstrap_section],
)
write_cache_manifest(manifest, CACHE_MANIFEST_DIR / f"5d_bootstrap_{universe_name}.json")
```

## 缓存失效条件

Per section:
- `results/features_{tier}.parquet` 内容变化
- `data/cache/prices/_manifest.json` 内容变化
- `data/cache/shares/_manifest.json` 内容变化（仅 mcap_buckets section）
- `universe_name` 参数变化
- 对应 section 函数源码变化

## 预期节省

- 跳过已完成 section：节省 per-section 运行时间（每个 section 1-5 分钟不等）
- 中断恢复：只需重跑被中断的那个 section，其他 10+ section 的结果在磁盘上不动
