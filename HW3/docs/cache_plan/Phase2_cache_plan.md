# Phase 2: Feature Engineering Cache

## 涉及文件

- `features/engineer.py`

## 问题

`build_features()` 在单个函数调用中完成所有子阶段（timestamps → row features → time-series →
PIT percentiles → momentum → stretch join），全部在内存中计算。如果 pipeline 在 Phase 2
被中断（或之后 Phase 出错需要重新运行），所有计算结果丢失。

当前 `engineer.py` 已经有 `--output` 参数写出最终 parquet，但没有内部 checkpoint，也没有
机制让 `run_all.py` 知道输出是否仍然有效。

## 方案

### 用 joblib.Memory 缓存 `build_features()` 输出

参考 `backtest/splits.py` 的 `ForwardReturnsCacheSignature` 模式：

```python
from joblib import Memory
from data.config import CACHE_MANIFEST_DIR

_engineer_memory = Memory(
    location=str(RESULTS_DIR / "cache" / "joblib" / "engineer"),
    verbose=0,
)


class EngineerCacheSignature:
    """所有影响 build_features() 输出的依赖。"""
    signals_path: str
    signals_sha256: str
    price_manifest_sha256: str
    tier: str
    include_momentum: bool
    sample_size: int          # --sample 参数
    source_hash: str          # build_features + 所有子阶段函数的源码哈希


def _build_engineer_source_hash() -> str:
    funcs = [build_features, _compute_timestamps, _compute_row_features,
             _compute_timeseries_features, _compute_pit_percentiles,
             _compute_momentum_features, _stream_stretch_join]
    source_blob = "\n\n".join(inspect.getsource(f) for f in funcs)
    return hashlib.sha256(source_blob.encode()).hexdigest()[:16]


@_engineer_memory.cache(ignore=["df"])
def _build_features_cached(
    signature: EngineerCacheSignature,
    df: pd.DataFrame,
    price_cache_dir: Path,
) -> pd.DataFrame:
    return build_features(df, price_cache_dir, signature.tier, signature.include_momentum)


def build_features_with_cache(
    df: pd.DataFrame,
    price_cache_dir: Path = PRICE_CACHE_DIR,
    tier: str = "enhanced",
    include_momentum: bool = True,
    sample_size: int = 0,
) -> pd.DataFrame:
    """带缓存的 build_features 入口。"""
    signature = _build_engineer_signature(
        signals_path=SIGNALS_PARQUET,
        price_cache_dir=price_cache_dir,
        tier=tier,
        include_momentum=include_momentum,
        sample_size=sample_size,
    )
    return _build_features_cached(signature, df, price_cache_dir)
```

### 修改 `main()`

- `build_features_with_cache()` 替代 `build_features()` 调用
- 写出 `--output` parquet 后，额外写出 cache manifest 到
  `CACHE_MANIFEST_DIR/2_{tier}.json`（给 `run_all.py` 的 `_skip()` 使用）

## 缓存失效条件

以下任一变化时缓存自动失效：
- `data/cache/signals.parquet` 文件内容变化（SHA256）
- `data/cache/prices/_manifest.json` 内容变化（影响 momentum 特征）
- `tier` 参数变化
- `include_momentum` 参数变化
- `--sample` 值变化
- `build_features()` 或其任一子阶段函数的源码变化

## 预期节省

- 每次跳过 Phase 2：~85 秒（enhanced）/ ~120 秒（stretch streaming）
