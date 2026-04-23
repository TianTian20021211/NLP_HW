from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.errors import error
from src.paths import (
    CACHE_DIR,
    CACHE_LOCK_PATH,
    GOLD_LABELED_PATH,
    GOLD_META_PATH,
    GOLD_SAMPLE_PATH,
    OLLAMA_FAIL_LOG,
    RUBRIC_PATH,
    SENTENCE_POOL_PATH,
)
from src.sentence_extraction import atomic_write_parquet, cache_dir_lock

# Default Ollama judge names match plan/README; override: gold_labeling label --models a,b,c
DEFAULT_OLLAMA_MODELS = ["llama3.1:8b", "qwen3:8b-q4_K_M", "gemma2:9b"]
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
# Default Haiku snapshot id; override: ANTHROPIC_HAIKU_MODEL env var
DEFAULT_HAIKU_MODEL = os.environ.get("ANTHROPIC_HAIKU_MODEL", "claude-haiku-4-5")

CONNECT_TIMEOUT_S = float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT_S = float(os.environ.get("OLLAMA_READ_TIMEOUT", "300"))
MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "5"))
FLUSH_EVERY = int(os.environ.get("OLLAMA_FLUSH_EVERY", "25"))

# Ollama persist-and-resume after retries exhausted (same as README); do not silently skip failed rows.
OLLAMA_EXHAUSTED_POLICY = (
    "On retry exhaustion, rows are written with parsed_label=null and ollama_error set; "
    "rows are retried on the next run. Failures are appended to cache/logs/ollama_failed_rows.jsonl."
)

# Merge/vote: missing parsed_label for a judge counts that vote as substantive in majority; writes j*_parse_fallback.
# Any disagreement among three normalized votes triggers Haiku; unanimous three-way vote sets gold_final with no API.
# Only one label/extract process should write cache/ at a time (see cache_dir_lock and README).
# API row estimate: haiku_api_usage_preview / notebooks/api_usage_check.ipynb (see README).


def safe_model_filename(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def labels_cache_path(model: str, labels_dir: Path = CACHE_DIR) -> Path:
    return Path(labels_dir) / f"labels_{safe_model_filename(model)}.parquet"


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_rubric(path: Path = RUBRIC_PATH) -> str:
    if not path.exists():
        error(f"Missing labeling rubric at {path}")
    return path.read_text(encoding="utf-8")


def build_user_prompt(text: str, rubric: str) -> str:
    return (
        "Classify the following earnings-call sentence.\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        f"SENTENCE:\n{text}\n\n"
        'Respond with JSON only: {"label":"boilerplate"} or {"label":"substantive"}.'
    )


def parse_json_label(raw: str) -> tuple[str | None, str | None]:
    raw = raw.strip()
    if not raw:
        return None, "empty_model_output"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not m:
            return None, "json_decode_error"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, "json_decode_error"
    label = obj.get("label")
    if label is None:
        return None, "missing_label_key"
    label_norm = str(label).strip().lower()
    if label_norm in {"boilerplate", "substantive"}:
        return label_norm, None
    return None, "invalid_label_value"


def _jitter_sleep(base: float, attempt: int) -> None:
    time.sleep(base * (2**attempt) + random.random() * 0.25)


def call_ollama_chat(
    model: str,
    user_prompt: str,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    session: requests.Session | None = None,
    post_fn: Callable[..., Any] | None = None,
) -> tuple[str | None, str | None]:
    """
    Call Ollama ``/api/chat`` with JSON format, min temperature, and ``think: false`` when supported.

    Returns ``(assistant_text, error_message)``. ``assistant_text`` is None on hard failure.
    """
    sess = session or requests.Session()
    post = post_fn or sess.post
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "format": "json",
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }
    last_err: str | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = post(
                url,
                json=payload,
                timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_err = f"{type(exc).__name__}:{exc}"
            append_jsonl(
                OLLAMA_FAIL_LOG,
                {
                    "ts": time.time(),
                    "model": model,
                    "attempt": attempt + 1,
                    "phase": "http",
                    "error": last_err,
                },
            )
            _jitter_sleep(0.75, attempt)
            continue
        if resp.status_code >= 500:
            last_err = f"http_{resp.status_code}"
            _jitter_sleep(0.75, attempt)
            continue
        if not resp.content:
            last_err = "empty_http_body"
            _jitter_sleep(0.75, attempt)
            continue
        try:
            data = resp.json()
        except json.JSONDecodeError:
            last_err = "ollama_response_not_json"
            _jitter_sleep(0.75, attempt)
            continue
        msg = (data.get("message") or {}) if isinstance(data, dict) else {}
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            last_err = "missing_message_content"
            _jitter_sleep(0.75, attempt)
            continue
        return content, None
    return None, last_err or "unknown_error"


def largest_remainder_quotas(weights: list[int], n: int) -> list[int]:
    weights_arr = np.asarray(weights, dtype=np.int64)
    total = int(weights_arr.sum())
    if total <= 0:
        return [0] * len(weights)
    exact = n * weights_arr.astype(np.float64) / float(total)
    floors = np.floor(exact).astype(np.int64)
    rem = int(n - int(floors.sum()))
    frac = exact - floors
    order = np.argsort(-frac)
    quotas = floors.copy()
    for k in range(max(rem, 0)):
        quotas[int(order[k % len(quotas)])] += 1
    return [int(x) for x in quotas.tolist()]


def stratified_sample_by_source_file(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample ``n`` rows stratified by ``source_file`` using largest-remainder quotas.

    If quotas exceed available rows in any stratum (rare when ``n`` approaches pool size),
    the deficit is filled uniformly at random from the remaining rows.
    """
    rng = np.random.default_rng(seed)
    if n > len(pool):
        error(f"gold_n={n} exceeds pool size {len(pool)}")
    vc = pool["source_file"].value_counts().sort_index()
    files = vc.index.to_numpy()
    weights = vc.to_numpy(dtype=np.int64)
    quotas = largest_remainder_quotas([int(x) for x in weights], n)
    chosen: list[int] = []
    for f, q in zip(files, quotas):
        positions = np.flatnonzero((pool["source_file"].to_numpy() == f))
        q = int(min(q, len(positions)))
        if q == 0:
            continue
        pick = rng.choice(positions, size=q, replace=False)
        chosen.extend([int(x) for x in pick.tolist()])
    chosen_arr = np.asarray(chosen, dtype=np.int64)
    if len(chosen_arr) < n:
        all_pos = np.arange(len(pool), dtype=np.int64)
        rest = np.setdiff1d(all_pos, chosen_arr, assume_unique=False)
        need = int(n - len(chosen_arr))
        extra = rng.choice(rest, size=need, replace=False)
        chosen_arr = np.concatenate([chosen_arr, extra])
    elif len(chosen_arr) > n:
        chosen_arr = rng.choice(chosen_arr, size=n, replace=False)
    out = pool.iloc[chosen_arr].copy()
    return out.reset_index(drop=True)


def gold_sample_meta_path(sample_path: Path) -> Path:
    return sample_path.with_suffix(sample_path.suffix + ".meta.json")


def try_load_cached_gold_sample(
    sample_path: Path,
    *,
    pool_path: Path,
    gold_n: int,
    seed: int,
) -> pd.DataFrame | None:
    meta_path = gold_sample_meta_path(sample_path)
    if not sample_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    pool_resolved = str(Path(pool_path).resolve())
    if str(meta.get("pool_path", "")) != pool_resolved:
        return None
    if int(meta.get("gold_n", -1)) != int(gold_n):
        return None
    if int(meta.get("random_seed", -1)) != int(seed):
        return None
    if not Path(pool_path).exists():
        return None
    try:
        pool_mtime = int(Path(pool_path).stat().st_mtime_ns)
    except OSError:
        return None
    if int(meta.get("pool_mtime_ns", -1)) != pool_mtime:
        return None
    return pd.read_parquet(sample_path)


def atomic_write_gold_sample(
    tbl: pd.DataFrame,
    out_path: Path,
    *,
    pool_path: Path,
    gold_n: int,
    seed: int,
) -> None:
    atomic_write_parquet(tbl, out_path)
    meta = {
        "gold_n": int(gold_n),
        "random_seed": int(seed),
        "rows_written": int(len(tbl)),
        "out": str(out_path.resolve()),
        "pool_path": str(Path(pool_path).resolve()),
        "pool_mtime_ns": int(Path(pool_path).stat().st_mtime_ns),
    }
    meta_out = gold_sample_meta_path(out_path)
    tmp = meta_out.with_suffix(meta_out.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, meta_out)


def sample_gold_table(
    pool: pd.DataFrame,
    gold_n: int,
    seed: int,
) -> pd.DataFrame:
    sampled = stratified_sample_by_source_file(pool, gold_n, seed)
    missing = {"sentence_id", "text", "source_file"} - set(sampled.columns)
    if missing:
        error(f"Pool is missing required columns: {sorted(missing)}")
    return sampled[["sentence_id", "text", "source_file"]].copy()


def load_completed_sentence_ids(labels_path: Path) -> set[str]:
    if not labels_path.exists():
        return set()
    df = pd.read_parquet(labels_path)
    if "sentence_id" in df.columns:
        df = df.drop_duplicates(subset=["sentence_id"], keep="last")
    if "sentence_id" not in df.columns:
        return set()
    if "parsed_label" not in df.columns:
        return set()
    ok = df["parsed_label"].notna() & (df["parsed_label"].astype(str).str.len() > 0)
    if "ollama_error" in df.columns:
        err_nonempty = df["ollama_error"].notna() & (df["ollama_error"].astype(str).str.len() > 0)
        ok = ok & ~err_nonempty
    return set(df.loc[ok, "sentence_id"].astype(str).tolist())


def run_ollama_labels_for_model(
    sample: pd.DataFrame,
    model: str,
    out_path: Path,
    *,
    rubric: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    session: requests.Session | None = None,
    post_fn: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """
    Label every row in ``sample`` with a single Ollama model, writing incremental Parquet checkpoints.

    Skips ``sentence_id`` rows that already have a successful ``parsed_label`` in ``out_path``.
    """
    rows: list[dict] = []
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if "sentence_id" in existing.columns:
            existing = existing.drop_duplicates(subset=["sentence_id"], keep="last")
        rows = existing.to_dict(orient="records")
    done = load_completed_sentence_ids(out_path)
    pending_mask = ~sample["sentence_id"].astype(str).isin(done)
    pending = sample.loc[pending_mask].reset_index(drop=True)
    if len(pending) == 0 and out_path.exists():
        return pd.read_parquet(out_path)
    successes_since_flush = 0
    for _, row in tqdm(
        pending.iterrows(),
        total=len(pending),
        desc=f"Ollama {model}",
        unit="sent",
    ):
        sid = str(row["sentence_id"])
        text = str(row["text"])
        prompt = build_user_prompt(text, rubric)
        raw, http_err = call_ollama_chat(model, prompt, base_url=base_url, session=session, post_fn=post_fn)
        parsed: str | None
        parse_err: str | None
        ollama_error: str | None
        if http_err:
            parsed, parse_err = None, None
            ollama_error = http_err
            append_jsonl(
                OLLAMA_FAIL_LOG,
                {
                    "ts": time.time(),
                    "sentence_id": sid,
                    "model": model,
                    "error": http_err,
                },
            )
        else:
            parsed, parse_err = parse_json_label(raw or "")
            ollama_error = parse_err
            if parse_err:
                append_jsonl(
                    OLLAMA_FAIL_LOG,
                    {
                        "ts": time.time(),
                        "sentence_id": sid,
                        "model": model,
                        "error": parse_err,
                        "raw_excerpt": (raw or "")[:500],
                    },
                )
        rows.append(
            {
                "sentence_id": sid,
                "text": text,
                "parsed_label": parsed,
                "raw_response": raw,
                "ollama_error": ollama_error,
            }
        )
        if http_err is None and parsed is not None:
            successes_since_flush += 1
        if successes_since_flush >= FLUSH_EVERY:
            atomic_write_parquet(pd.DataFrame(rows), out_path)
            successes_since_flush = 0
    atomic_write_parquet(pd.DataFrame(rows), out_path)
    return pd.read_parquet(out_path)


def merge_judge_tables(sample: pd.DataFrame, model_names: list[str], labels_dir: Path) -> pd.DataFrame:
    """
    Join cached judge outputs onto the gold sample.

    Every ``sentence_id`` in ``sample`` must exist in each judge cache. If ``parsed_label`` is
    missing, the judge vote falls back to **substantive** (assignment guidance / recall-oriented
    default), recorded in ``j{i}_parse_fallback``.
    """
    out = sample[["sentence_id", "text", "source_file"]].copy()
    for i, model in enumerate(model_names):
        p = labels_cache_path(model, labels_dir=labels_dir)
        if not p.exists():
            error(f"Missing judge cache for model {model}: {p}")
        df = pd.read_parquet(p)
        if "sentence_id" not in df.columns:
            error(f"Invalid judge cache (missing sentence_id): {p}")
        df = df.drop_duplicates(subset=["sentence_id"], keep="last").set_index("sentence_id", drop=False)
        key = f"j{i+1}"
        labs: list[str] = []
        raws: list[str | None] = []
        fallbacks: list[bool] = []
        for sid in out["sentence_id"].astype(str).tolist():
            if sid not in df.index:
                error(
                    f"Refusing to merge: sentence_id missing in {model} cache ({p.name}): {sid}. "
                    "Finish that model's Ollama pass or regenerate caches."
                )
            row = df.loc[sid]
            pl = row.get("parsed_label")
            raw = row.get("raw_response")
            if pd.notna(pl) and str(pl).strip():
                labs.append(str(pl).strip().lower())
                fallbacks.append(False)
            else:
                labs.append("substantive")
                fallbacks.append(True)
            if pd.isna(raw):
                raws.append(None)
            else:
                raws.append(str(raw))
        out[f"{key}_label"] = labs
        out[f"{key}_raw"] = raws
        out[f"{key}_parse_fallback"] = fallbacks
    return out


def majority_label(a: str, b: str, c: str) -> tuple[str | None, str | None, bool]:
    """
    Return ``(local_majority_guess, discord_tag, needs_haiku_api)``.

    ``needs_haiku_api`` is true whenever the three normalized votes are **not** all identical
    (including missing/empty cells). Unanimous rows use ``local_majority_guess`` as ``gold_final``
    without calling Anthropic; any disagreement is resolved by Haiku in ``merge_vote_and_arbitrate``.
    """
    labels = [a, b, c]
    if any(pd.isna(x) or str(x) == "" for x in labels):
        return None, "missing_judge_label", True
    s = [str(x).lower() for x in labels]
    if s[0] == s[1] == s[2]:
        return s[0], "3-0", False
    if s[0] == s[1] or s[0] == s[2]:
        return s[0], "2-1", True
    if s[1] == s[2]:
        return s[1], "2-1", True
    return None, "1-1-1", True


def attach_vote_preview(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``vote_discord``, ``needs_haiku_api``, and ``vote_is_3_0`` columns using the same rules as
    ``majority_label`` / ``merge_vote_and_arbitrate`` (without calling Anthropic).

    ``needs_haiku_api`` is true for any row where the three judge labels are not all equal after
    normalization (including missing/empty), i.e. **any** inter-model disagreement triggers API.
    """
    required = ["j1_label", "j2_label", "j3_label"]
    miss = [c for c in required if c not in merged.columns]
    if miss:
        error(f"attach_vote_preview: missing columns {miss}")
    j1, j2, j3 = merged["j1_label"], merged["j2_label"], merged["j3_label"]
    na_lab = j1.isna() | j2.isna() | j3.isna()
    a = j1.astype(str).str.strip().str.lower()
    b = j2.astype(str).str.strip().str.lower()
    c = j3.astype(str).str.strip().str.lower()
    empty = (~na_lab) & ((a == "") | (b == "") | (c == ""))
    missing = na_lab | empty
    three_zero = (a == b) & (b == c) & (~missing)
    split_three = (a != b) & (b != c) & (a != c) & (~missing)
    two_one = (~missing) & (~three_zero) & (~split_three)
    discord = np.select(
        [missing, split_three, three_zero, two_one],
        ["missing_judge_label", "1-1-1", "3-0", "2-1"],
        default="unknown",
    )
    needs_haiku_api = missing | (~three_zero)
    out = merged.copy()
    out["vote_discord"] = discord
    out["needs_haiku_api"] = needs_haiku_api
    out["vote_is_3_0"] = three_zero & (~missing)
    out["any_judge_parse_fallback"] = np.zeros(len(out), dtype=bool)
    for col in ("j1_parse_fallback", "j2_parse_fallback", "j3_parse_fallback"):
        if col in out.columns:
            out["any_judge_parse_fallback"] = out["any_judge_parse_fallback"] | out[col].astype(bool).to_numpy()
    return out


def haiku_api_usage_preview(
    sample: pd.DataFrame,
    model_names: list[str],
    labels_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """
    Join cached Ollama judge tables onto ``sample`` and report how many rows would invoke Haiku.

    This mirrors the voting gate inside ``merge_vote_and_arbitrate`` but performs **no** API
    calls. Use it to decide whether to set API keys / run ``merge-vote``.

    ``needs_haiku_api_rows`` counts rows where the three judge labels are **not** all the same
    (after merge-time normalization), including ``2-1`` and ``1-1-1`` patterns and missing labels.
    """
    merged = merge_judge_tables(sample, model_names, labels_dir=labels_dir)
    merged = attach_vote_preview(merged)
    n = int(len(merged))
    n_api = int(merged["needs_haiku_api"].sum())
    n_30 = int(merged["vote_is_3_0"].sum())
    n_11 = int((merged["vote_discord"].astype(str) == "1-1-1").sum())
    n_21 = int((merged["vote_discord"].astype(str) == "2-1").sum())
    n_miss = int((merged["vote_discord"].astype(str) == "missing_judge_label").sum())
    n_fb = int(merged["any_judge_parse_fallback"].sum()) if "any_judge_parse_fallback" in merged.columns else 0
    summary: dict[str, float | int | bool] = {
        "rows": n,
        "needs_haiku_api_rows": n_api,
        "discord_3_0_rows": n_30,
        "discord_2_1_rows": n_21,
        "discord_1_1_1_rows": n_11,
        "discord_missing_judge_label_rows": n_miss,
        "rows_with_parse_fallback": n_fb,
        "frac_needs_haiku_api": float(n_api / n) if n else 0.0,
    }
    return merged, summary


def call_anthropic_haiku_json(
    text: str,
    rubric: str,
    *,
    model: str = DEFAULT_HAIKU_MODEL,
    client: Any | None = None,
) -> tuple[str | None, str | None]:
    try:
        import anthropic
    except ImportError as exc:
        return None, f"anthropic_import_error:{exc}"
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if client is None and not api_key:
        return None, "missing_ANTHROPIC_API_KEY"
    cli = client or anthropic.Anthropic(api_key=api_key)
    user_prompt = build_user_prompt(text, rubric)
    try:
        msg = cli.messages.create(
            model=model,
            max_tokens=128,
            temperature=0,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as exc:
        return None, f"anthropic_error:{type(exc).__name__}:{exc}"
    parts = getattr(msg, "content", []) or []
    chunks: list[str] = []
    for p in parts:
        txt = getattr(p, "text", None)
        if isinstance(txt, str):
            chunks.append(txt)
    raw = "\n".join(chunks).strip()
    label, err = parse_json_label(raw)
    if err:
        return None, err
    return label, None


def merge_vote_and_arbitrate(
    sample: pd.DataFrame,
    model_names: list[str],
    *,
    rubric: str,
    labels_dir: Path = CACHE_DIR,
    anthropic_client: Any | None = None,
) -> pd.DataFrame:
    keys = [f"j{i+1}" for i in range(len(model_names))]
    merged = merge_judge_tables(sample, model_names, labels_dir=labels_dir)
    gold_final: list[str] = []
    discord: list[str] = []
    api_used: list[bool] = []
    api_err: list[str | None] = []
    tie_break_label: list[str | None] = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Vote/Haiku", unit="row"):
        labs = [str(row[f"{k}_label"]) for k in keys]
        final, disc, needs_api = majority_label(labs[0], labs[1], labs[2])
        used = False
        err: str | None = None
        arb: str | None = None
        if needs_api:
            label, err = call_anthropic_haiku_json(str(row["text"]), rubric, client=anthropic_client)
            used = True
            if label is None:
                final = "substantive"
                arb = None
                if err is None:
                    err = "unknown_api_failure"
            else:
                final = label
                arb = label
        gold_final.append(str(final))
        discord.append(str(disc))
        api_used.append(bool(used))
        api_err.append(err)
        tie_break_label.append(arb)
    merged = merged.copy()
    merged["gold_final"] = gold_final
    merged["discord"] = discord
    merged["api_used"] = api_used
    merged["api_error"] = api_err
    merged["tie_break_label"] = tie_break_label
    return merged


def disagreement_summary(merged: pd.DataFrame, model_names: list[str]) -> dict[str, Any]:
    keys = [f"j{i+1}_label" for i in range(len(model_names))]
    if len(keys) != 3:
        return {}
    a, b, c = merged[keys[0]].astype(str), merged[keys[1]].astype(str), merged[keys[2]].astype(str)
    all_three = (a == b) & (b == c)
    two_way_ab = (a == b).mean()
    two_way_ac = (a == c).mean()
    two_way_bc = (b == c).mean()
    no_majority = (merged["discord"].astype(str) == "1-1-1").mean()
    frac_two_one = (merged["discord"].astype(str) == "2-1").mean()
    any_disagreement = (~all_three).astype(float).mean()
    return {
        "frac_all_three_agree": float(all_three.mean()),
        "frac_judges_not_unanimous": float(any_disagreement),
        "pairwise_agree_ab": float(two_way_ab),
        "pairwise_agree_ac": float(two_way_ac),
        "pairwise_agree_bc": float(two_way_bc),
        "frac_discord_2_1": float(frac_two_one),
        "frac_1_1_1": float(no_majority),
        "api_calls": int(merged["api_used"].sum()),
    }


@dataclass(frozen=True)
class GoldRunConfig:
    gold_n: int
    seed: int
    models: tuple[str, ...]
    ollama_url: str


def write_gold_meta(
    path: Path,
    *,
    cfg: GoldRunConfig,
    summary: dict[str, Any],
) -> None:
    payload = {
        "gold_n": cfg.gold_n,
        "random_seed": cfg.seed,
        "ollama_models": list(cfg.models),
        "ollama_url": cfg.ollama_url,
        "haiku_model": DEFAULT_HAIKU_MODEL,
        "ollama_exhausted_policy": OLLAMA_EXHAUSTED_POLICY,
        "disagreement_summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def cmd_sample(args: argparse.Namespace) -> None:
    pool_path = Path(args.pool)
    pool = pd.read_parquet(pool_path)
    gold_n = int(args.cap) if args.cap is not None else int(args.gold_n)
    out = Path(args.out)
    seed = int(args.seed)
    if not args.force:
        cached = try_load_cached_gold_sample(out, pool_path=pool_path, gold_n=gold_n, seed=seed)
        if cached is not None:
            print(f"Loaded {len(cached)} gold sample rows from cache {out.resolve()}")
            return
    tbl = sample_gold_table(pool, gold_n, seed)
    atomic_write_gold_sample(tbl, out, pool_path=pool_path, gold_n=gold_n, seed=seed)
    print(f"Wrote {len(tbl)} gold sample rows to {out.resolve()}")


def cmd_label(args: argparse.Namespace) -> None:
    rubric = load_rubric(Path(args.rubric))
    sample = pd.read_parquet(Path(args.sample))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        error("--models must list at least one Ollama model")
    with cache_dir_lock(Path(args.lock_path), not args.no_cache_lock):
        for model in models:
            out_path = labels_cache_path(model, labels_dir=Path(args.labels_dir))
            run_ollama_labels_for_model(
                sample,
                model,
                out_path,
                rubric=rubric,
                base_url=args.ollama_url,
            )
    print("Finished Ollama labeling pass(es).")


def cmd_merge_vote(args: argparse.Namespace) -> None:
    rubric = load_rubric(Path(args.rubric))
    sample = pd.read_parquet(Path(args.sample))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    merged = merge_vote_and_arbitrate(
        sample,
        models,
        rubric=rubric,
        labels_dir=Path(args.labels_dir),
    )
    out = Path(args.out)
    atomic_write_parquet(merged, out)
    cfg = GoldRunConfig(
        gold_n=int(args.gold_n),
        seed=int(args.seed),
        models=tuple(models),
        ollama_url=str(args.ollama_url),
    )
    summ = disagreement_summary(merged, models)
    write_gold_meta(Path(args.meta_out), cfg=cfg, summary=summ)
    print(f"Wrote gold labels to {out.resolve()}")
    print(json.dumps(summ, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gold labeling pipeline (sample → Ollama judges → merge/vote/Haiku).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="Stratified sample gold_n rows from the sentence pool.")
    s.add_argument("--pool", type=Path, default=SENTENCE_POOL_PATH)
    s.add_argument("--out", type=Path, default=GOLD_SAMPLE_PATH)
    s.add_argument("--gold-n", type=int, default=3000)
    s.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Shortcut for --gold-n (smoke tests).",
    )
    s.add_argument("--seed", type=int, default=42)
    s.add_argument(
        "--force",
        action="store_true",
        help="Resample even if a matching gold_sample.parquet cache exists.",
    )
    s.set_defaults(func=cmd_sample)

    l = sub.add_parser("label", help="Run Ollama judges sequentially (one model at a time).")
    l.add_argument("--sample", type=Path, default=GOLD_SAMPLE_PATH)
    l.add_argument("--rubric", type=Path, default=RUBRIC_PATH)
    l.add_argument("--models", type=str, default=",".join(DEFAULT_OLLAMA_MODELS))
    l.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    l.add_argument("--labels-dir", type=Path, default=CACHE_DIR)
    l.add_argument("--lock-path", type=Path, default=CACHE_LOCK_PATH)
    l.add_argument("--no-cache-lock", action="store_true")
    l.set_defaults(func=cmd_label)

    m = sub.add_parser("merge-vote", help="Join judge caches, validate completeness, vote, arbitrate ties.")
    m.add_argument("--sample", type=Path, default=GOLD_SAMPLE_PATH)
    m.add_argument("--rubric", type=Path, default=RUBRIC_PATH)
    m.add_argument("--models", type=str, default=",".join(DEFAULT_OLLAMA_MODELS))
    m.add_argument("--out", type=Path, default=GOLD_LABELED_PATH)
    m.add_argument("--meta-out", type=Path, default=GOLD_META_PATH)
    m.add_argument("--labels-dir", type=Path, default=CACHE_DIR)
    m.add_argument("--gold-n", type=int, default=3000)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--ollama-url", type=str, default=DEFAULT_OLLAMA_URL)
    m.set_defaults(func=cmd_merge_vote)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
