# Phase 0: Shared Cache Infrastructure

## 涉及文件

- `data/cache_utils.py` — **新建**，共享缓存工具
- `data/config.py` — 添加 cache manifest 目录路径
- `run_all.py` — 替换 `_skip()` 为内容感知版本

## 问题

`run_all.py:100-108` 的 `_skip()` 函数硬编码跳过 Phase 2+：

```python
def _skip(artifact_key: str, force: bool) -> bool:
    if force:
        return False
    if not artifact_key.startswith("1."):   # 只有 Phase 1
        return False
    path = ARTIFACTS.get(artifact_key)
    return _artifact_exists(path)
```

当前设计理由（注释 103-106 行）："Later phases are matrix jobs where one generic artifact
can hide missing tier/universe/model/cadence outputs"。这是错误的——每组合独立缓存即可。

## 方案

### 1. 新建 `data/cache_utils.py`

提供轻量级 cache manifest 机制，供所有 Phase 共用：

```python
import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

CACHE_MANIFEST_VERSION = 1

@dataclass
class CacheManifest:
    cache_version: int
    computed_at: str
    phase: str
    parameters: dict          # {tier, model, universe, horizon, cadence, ...}
    input_hashes: dict        # {rel_path: sha256}
    source_hash: str          # SHA256 of function source code


def build_cache_manifest(
    phase: str,
    parameters: dict,
    input_paths: list[Path],
    source_funcs: list,
) -> CacheManifest:
    """对所有输入文件和函数源码做 SHA256，构建 manifest。"""
    input_hashes = {}
    for p in input_paths:
        if p.exists():
            input_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            input_hashes[str(p)] = ""
    source_blob = "\n\n".join(inspect.getsource(f) for f in source_funcs)
    source_hash = hashlib.sha256(source_blob.encode()).hexdigest()[:16]
    return CacheManifest(
        cache_version=CACHE_MANIFEST_VERSION,
        computed_at=datetime.now(timezone.utc).isoformat(),
        phase=phase,
        parameters=parameters,
        input_hashes=input_hashes,
        source_hash=source_hash,
    )


def write_cache_manifest(manifest: CacheManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2))


def validate_cache_manifest(manifest_path: Path) -> bool:
    """读取 manifest 并验证所有输入文件哈希是否仍然匹配。"""
    if not manifest_path.exists():
        return False
    try:
        data = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, KeyError):
        return False
    if data.get("cache_version") != CACHE_MANIFEST_VERSION:
        return False
    for path_str, expected_hash in data["input_hashes"].items():
        p = Path(path_str)
        if not p.exists():
            return False
        current_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        if current_hash != expected_hash:
            return False
    return True
```

### 2. `data/config.py` 添加路径

```python
CACHE_MANIFEST_DIR: Path = RESULTS_DIR / "cache" / "manifests"
```

### 3. `run_all.py` 改造 `_skip()`

```python
from data.cache_utils import validate_cache_manifest

# ARTIFACTS 需要改成 output_path -> manifest_path 的映射，或者保持简单格式
# 对于 Phase 2+，检查输出文件存在 + cache manifest 有效

def _skip(artifact_key: str, force: bool) -> bool:
    if force:
        return False
    path = ARTIFACTS.get(artifact_key)
    if path is None:
        return False
    # Phase 1: 简单存在检查（price/shares 额外做 manifest coverage 检查）
    if artifact_key.startswith("1."):
        return _artifact_exists(path)
    # Phase 2+: 内容感知缓存
    manifest_path = CACHE_MANIFEST_DIR / f"{artifact_key}.json"
    return _artifact_exists(path) and validate_cache_manifest(manifest_path)
```

## 与各 Phase 的集成

每个 Phase 模块在完成计算后调用：

```python
from data.cache_utils import build_cache_manifest, write_cache_manifest

manifest = build_cache_manifest(
    phase="2_enhanced",
    parameters={"tier": "enhanced"},
    input_paths=[signals_path, price_manifest],
    source_funcs=[build_features, _compute_timestamps, ...],
)
write_cache_manifest(manifest, CACHE_MANIFEST_DIR / "2_enhanced.json")
```

`run_all.py` 在调用模块前用 `validate_cache_manifest()` 判断是否跳过。
