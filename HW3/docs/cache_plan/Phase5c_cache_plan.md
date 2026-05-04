# Phase 5c: Portfolio Simulation Cache

## 涉及文件

- `backtest/portfolio.py`
- `backtest/portfolio_batch.py`

## 问题

`PortfolioSimulator.run()` 按交易日遍历整个回测期（~2500+ 天），逐日：
1. 查询 pending signals（lookback 窗口内）
2. 获取 prices + shares 数据
3. 计算权重 + 执行交易
4. 计算每日 P&L

`PortfolioSimulator._market_cache` 是一个内存 dict（近距离矩阵、交易日历、再平衡日期），
从不持久化到磁盘。所有日度 P&L 和权重只存在于 `PortfolioResult` 对象中，中断后全部丢失。

Portfolio batch runner (`portfolio_batch.py`) 使用 `ProcessPoolExecutor` 并行提交多个
模拟任务，但各子进程的结果通过内存传回主进程，无中间持久化。

## 方案

### 整体缓存

`PortfolioSimulator.run()` 的确定性很高——相同 signals + 相同参数 → 相同结果。
适合做整体缓存。

在 CLI `main()` 中，写出最终输出后，额外写出 cache manifest：

```python
from data.cache_utils import build_cache_manifest, write_cache_manifest

def main():
    # ... existing simulation ...
    result = simulator.run(signals, cadence, lookback, ...)
    _persist_portfolio_results(result, output_dir, suffix, ...)

    # 写出 cache manifest
    manifest = build_cache_manifest(
        phase=f"5c_{suffix}",   # e.g. "5c_sp500_ridge_enhanced_predh5d_weekly_5d"
        parameters={
            "universe": args.universe,
            "cadence": args.cadence,
            "lookback": args.lookback,
            "long_frac": args.long_frac,
            "cost_bps": args.cost_bps,
            "weekly_day": args.weekly_day,
            "tag": args.tag,
        },
        input_paths=[
            args.signals,
            args.features,
            PRICE_CACHE_DIR / "_manifest.json",
            UNIVERSE_CACHE_DIR / f"{args.universe}_pit.parquet",
        ],
        source_funcs=[PortfolioSimulator.run, PortfolioSimulator._rebalance,
                      PortfolioSimulator._compute_daily_pnl],
    )
    write_cache_manifest(manifest, CACHE_MANIFEST_DIR / f"5c_{suffix}.json")
```

### 参数空间

Portfolio 是参数组合最多的阶段：

| 参数 | 值 |
|---|---|
| universe | sp500, sp1500, ru3k |
| model/信号源 | ridge_enhanced_predh5d, ... |
| cadence | daily, weekly, monthly |
| lookback | 5, 10, 21 |
| long_frac | 0.1, 0.2 |
| cost_bps | 3, 5, 7, 10 |
| weekly_day | monday, friday |

每个组合有独立输出文件（suffix 编码了所有参数），每个组合有独立 manifest。

### `run_all.py` 处理

```python
# Portfolio 按 suffix 拆分
for universe in ["sp500", "sp1500", "ru3k"]:
    for cadence in ["weekly"]:
        key = f"5c_{universe}_ridge_enhanced_predh5d_{cadence}_5d"
        if _skip(key, args.force):
            continue
        cmd = ["python", "-m", "backtest.portfolio", ...]
```

## 缓存失效条件

- Signals parquet 内容变化
- Features parquet 内容变化
- `data/cache/prices/_manifest.json` 内容变化
- Universe PIT parquet 内容变化
- 任一参数值变化（cadence, lookback, long_frac, cost_bps, weekly_day）
- `PortfolioSimulator` 相关函数源码变化

## 预期节省

- 每次跳过 per-universe per-参数组合：节省 ~5-15 分钟（取决于 universe 规模）
- 中断恢复：已完成组合不动，只重跑被打断的那个
