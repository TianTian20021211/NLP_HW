# Phase 4: Walk-Forward Backtest Cache

## 涉及文件

- `backtest/model.py`

## 问题

`run_walk_forward()` 是 pipeline 中运行时间最长的阶段，也是中断损失最大的阶段：

1. **Trained model 从不序列化保存**：每个 fold 的 Ridge/LightGBM/XGBoost 模型
   拟合后被用于预测，然后丢弃。中断后所有 fold 必须重跑。
2. **Per-fold 无 checkpoint**：26 folds × 3 models × 5 horizons = 390 个独立计算单元，
   但只有全部完成后才写出 `oos_pred_*_{horizon}d.parquet`。
3. **每 horizon 独立循环**：OOS 预测是 per-horizon 写出的（`oos_pred_*_h5d.parquet` 等），
   但 `run_all.py` 不检查部分完成状态就重新调用整个模块。

如果 pipeline 在第 20 个 fold 被杀掉（OOM、超时、手动 Ctrl+C），所有 19 个已完成 fold
的模型和预测结果全部丢失。

## 方案

### 策略：per-horizon cache manifest + per-fold 增量 checkpoint

#### 1. Per-horizon cache manifest

`run_walk_forward()` 按 horizon 循环（1, 3, 5, 10, 20）。每个 horizon 独立写出输出文件：

```
results/audit/oos_pred_{model}_{tier}_{universe}_{signal_type}_h{horizon}d.parquet
```

在 `run_all.py` 的 `ARTIFACTS` 中，将 Phase 4 按 horizon 拆分：

```python
ARTIFACTS = {
    ...
    "4_ridge_enhanced_sp500_h5d": AUDIT_DIR / "oos_pred_ridge_enhanced_sp500_total_h5d.parquet",
    "4_ridge_enhanced_sp500_h20d": AUDIT_DIR / "oos_pred_ridge_enhanced_sp500_total_h20d.parquet",
    ...
}
```

每个 horizon 完成后写出独立 manifest：

```python
manifest = build_cache_manifest(
    phase=f"4_{model}_{tier}_{universe}_h{horizon}d",
    parameters={...},
    input_paths=[features_path, hparams_json, price_manifest],
    source_funcs=[run_walk_forward, _run_single_fold, ...],
)
write_cache_manifest(manifest, CACHE_MANIFEST_DIR / f"4_{model}_{tier}_{universe}_h{horizon}d.json")
```

#### 2. Per-fold model checkpointing

在 `run_walk_forward()` 的 fold 循环内部，每个 fold 完成后：

```python
MODEL_CACHE_DIR = RESULTS_DIR / "cache" / "models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

for fold_idx, (train_df, test_df) in enumerate(folds):
    fold_model_path = MODEL_CACHE_DIR / f"{model}_{tier}_{universe}_h{horizon}d_fold{fold_idx}.pkl"

    if fold_model_path.exists():
        # 跳过已完成的 fold，加载已有预测
        cached_pred = pd.read_parquet(
            MODEL_CACHE_DIR / f"{model}_{tier}_{universe}_h{horizon}d_fold{fold_idx}_pred.parquet"
        )
        oos_predictions.append(cached_pred)
        continue

    # 正常训练
    model = train_fold(train_df)
    pred = predict_fold(model, test_df)

    # 序列化 model 和 prediction
    import pickle
    with open(fold_model_path, "wb") as f:
        pickle.dump(model, f)
    pred.to_parquet(
        MODEL_CACHE_DIR / f"{model}_{tier}_{universe}_h{horizon}d_fold{fold_idx}_pred.parquet"
    )

    oos_predictions.append(pred)
```

#### 3. 完整 horizon 完成后清理 per-fold 文件

当所有 fold 完成、OOS 预测已聚合写出后，删除 per-fold 中间文件：

```python
# horizon 完成后
for fold_idx in range(len(folds)):
    (MODEL_CACHE_DIR / f"{model}_{tier}_{universe}_h{horizon}d_fold{fold_idx}.pkl").unlink(missing_ok=True)
    (MODEL_CACHE_DIR / f"{model}_{tier}_{universe}_h{horizon}d_fold{fold_idx}_pred.parquet").unlink(missing_ok=True)
```

### `run_all.py` 改造

Phase 4 不再是一个大步骤，而是按 model × horizon 拆分：

```python
def phase4(args):
    for model in ["ridge", "lightgbm", "xgboost"]:
        for h in [1, 3, 5, 10, 20]:
            key = f"4_{model}_enhanced_sp500_h{h}d"
            if _skip(key, args.force):
                print(f"  skip {key} (cached)")
                continue
            cmd = ["python", "-m", "backtest.model",
                   "--model", model,
                   "--horizons", str(h),   # 单 horizon
                   ...]
            if not _run(cmd, f"4 Walk-forward {model} h{h}d"):
                return False
    return True
```

好处：即使 pipeline 在 `lightgbm h10d` 被杀掉，已完成的所有 horizon × model 组合都保留。

## 缓存失效条件

- `results/features_{tier}.parquet` 内容变化
- `results/hparams/{tier}/h{horizon}d/frozen_hparams_{model}.json` 内容变化
- `data/cache/prices/_manifest.json` 内容变化
- `model_name`、`tier`、`horizon`、`universe_name`、`signal_type` 参数变化
- `run_walk_forward()` 及 fold 相关函数源码变化

## 预期节省

- 每次跳过已完成 horizon：节省 26 folds 的 train + predict 时间
- 断点续跑：只重跑未完成的 fold，已完成的直接加载
- 典型场景：3 models × 5 horizons = 15 个独立缓存单元，中断后只重跑被打断的那个
