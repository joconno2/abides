# -*- coding: utf-8 -*-
"""CMA-ES harness for tuning the adaptive market maker under ABIDES."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import multiprocessing
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cma
except Exception as exc:  # pragma: no cover
    raise RuntimeError("CMA library is required: pip install cma") from exc

import mm_config as cfg
from mmcore.datasets import (
    auto_dataset_split,
    make_split_combos,
    prepare_combo_plan,
    resolve_dataset_paths,
)
from mmcore.events import (
    EventBus,
    EvaluationCompleteEvent,
    EvaluationResult,
    EvaluationStartEvent,
    GenerationSummaryEvent,
    RunCompleteEvent,
)

EVENT_QUEUE = None


def _worker_init(event_queue):  # pragma: no cover - worker side init
    global EVENT_QUEUE
    EVENT_QUEUE = event_queue

ROOT = cfg.ROOT
ABIDES_CORE = cfg.ABIDES_CORE
RUNS_ROOT = cfg.RUNS_ROOT
RUN_LABEL = datetime.now(timezone.utc).strftime("mm_%Y%m%dT%H%M%SZ")
RUNS = RUNS_ROOT / RUN_LABEL
HISTORIC_RUNS = cfg.HISTORIC_RUNS
EVALS = RUNS / "evals"
for path in (RUNS_ROOT, RUNS, EVALS, HISTORIC_RUNS):
    path.mkdir(parents=True, exist_ok=True)
ABIDES_PY = str((ABIDES_CORE / "abides.py").resolve())

DEFAULT_LOB_FILE = cfg.DEFAULT_LOB_FILE
DEFAULT_LOB_DIR = cfg.DEFAULT_LOB_DIR
DEFAULT_SECONDS = cfg.DEFAULT_SECONDS
DEFAULT_TIMEOUT = cfg.DEFAULT_TIMEOUT
DEFAULT_MINUTES = cfg.DEFAULT_MINUTES
DEFAULT_POPSIZE = cfg.DEFAULT_POPSIZE
DEFAULT_GENS = cfg.DEFAULT_GENS
DEFAULT_MAX_EVALS = cfg.DEFAULT_MAX_EVALS
DEFAULT_EARLY_STOP = cfg.DEFAULT_EARLY_STOP
DEFAULT_SEEDS = cfg.DEFAULT_SEEDS
DEFAULT_DAYS = cfg.DEFAULT_DAYS
DEFAULT_TRAIN_SEEDS = cfg.DEFAULT_TRAIN_SEEDS
DEFAULT_VAL_SEEDS = cfg.DEFAULT_VAL_SEEDS
DEFAULT_TEST_SEEDS = cfg.DEFAULT_TEST_SEEDS
DEFAULT_TRAIN_DAYS = cfg.DEFAULT_TRAIN_DAYS
DEFAULT_VAL_DAYS = cfg.DEFAULT_VAL_DAYS
DEFAULT_TEST_DAYS = cfg.DEFAULT_TEST_DAYS
TRAIN_DATASET_COUNT = cfg.TRAIN_DATASET_COUNT
VAL_DATASET_COUNT = cfg.VAL_DATASET_COUNT
TEST_DATASET_COUNT = cfg.TEST_DATASET_COUNT
WARMUP_GENERATIONS = cfg.WARMUP_GENERATIONS
WARMUP_DATASET_AGG = cfg.WARMUP_DATASET_AGG
FULL_DATASET_AGG = cfg.FULL_DATASET_AGG
DATASET_SPLIT_SEED = cfg.DATASET_SPLIT_SEED
DEFAULT_INV_PENALTY = cfg.DEFAULT_INV_PENALTY
DEFAULT_DRAWDOWN_THRESHOLD = cfg.DEFAULT_DRAWDOWN_THRESHOLD
DEFAULT_DRAWDOWN_PENALTY = cfg.DEFAULT_DRAWDOWN_PENALTY
DEFAULT_DRAWDOWN_CLIP = cfg.DEFAULT_DRAWDOWN_CLIP
DEFAULT_SCORE_TRIM = cfg.DEFAULT_SCORE_TRIM
DEFAULT_START_CASH = cfg.DEFAULT_START_CASH
DEFAULT_ROBUST_QUANTILE = cfg.DEFAULT_ROBUST_QUANTILE
DEFAULT_THIN = cfg.DEFAULT_THIN

ANSI_RESET = cfg.ANSI_RESET
ANSI_GREEN = cfg.ANSI_GREEN
ANSI_RED = cfg.ANSI_RED
ANSI_YELLOW = cfg.ANSI_YELLOW
ANSI_BLUE = cfg.ANSI_BLUE

# ---------- genome & mapping ----------
# Genome dims (we keep 6 for continuity, but map to *many* agent attrs):
# 0: participation proxy  → pov in [0.01, 0.50]
# 1: quote size proxy     → min/order/quote size
# 2: skew proxy           → skew_beta (inventory skew)
# 3: ladder spacing proxy → level_spacing & window sizing
# 4: smoothing proxy      → spread_alpha
# 5: cancellation proxy   → cancel_limit_delay (ns)
# 6: inventory/risk proxy → inventory_risk_aversion
# 7: inventory multiplier → max inventory scaling
BOUNDS = np.array([
    [  1,   30],   # 0 → pov proxy (0.01–0.35)
    [ 50,  800],   # 1 → size (tighter to avoid runaway inventory)
    [  0,    6],   # 2 → skew beta (limit extreme skew)
    [  6,   15],   # 3 → level spacing
    [0.10, 0.85],  # 4 → spread alpha (avoid ultraslow/ultrafast updates)
    [ 20,  250],   # 5 → cancel delay (ns)
    [0.10,  2.0],  # 6 → inventory risk aversion
    [1.00, 3.0],   # 7 → max inventory multiplier
], dtype=float)
MID = BOUNDS.mean(axis=1)
SIGMA0 = float((BOUNDS[:,1] - BOUNDS[:,0]).mean() / 3.0)

def _clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))


def _use_color(args):
    if getattr(args, "no_color", False):
        return False
    return sys.stdout.isatty()


def _fmt_score(score, args):
    if not _use_color(args):
        return f"{score:.6f}"
    color = ANSI_GREEN if score > 0 else ANSI_RED if score < 0 else ANSI_YELLOW
    return f"{color}{score:.6f}{ANSI_RESET}"


def _progress_bar(current, total, width=24):
    if total <= 0:
        return "[?]"
    pct = min(1.0, max(0.0, current / total))
    filled = int(round(pct * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({pct * 100:5.1f}%)"


def _split_tag(split: str) -> str:
    split = (split or "train").lower()
    if split.startswith("train"):
        return "tr"
    if split.startswith("val"):
        return "va"
    if split.startswith("test"):
        return "te"
    return split[:2]


def _cvar(scores, quantile, trim_frac):
    if not scores:
        return None
    arr = np.asarray([float(s) for s in scores if s is not None], dtype=float)
    if arr.size == 0:
        return None
    arr = np.sort(arr)
    trim = int(arr.size * trim_frac)
    if trim > 0 and trim * 2 < arr.size:
        arr = arr[trim:-trim]
    if arr.size == 0:
        return None
    if quantile is None or quantile <= 0.0:
        return float(arr.mean())
    worst_count = max(1, int(math.ceil(arr.size * quantile)))
    return float(arr[:worst_count].mean())


def _make_eval_paths(gen, idx, combo, base_dir=None):
    split = combo.get("split", "train")
    split_tag = _split_tag(split)
    day = combo.get("day")
    seed = combo.get("seed")
    label = f"mm_mvp_mm_g{gen:03d}_i{idx:02d}_{split_tag}_d{day}_s{seed}"
    base = Path(base_dir) if base_dir else EVALS
    out_dir = base / f"g{gen:03d}_i{idx:02d}_{split_tag}_d{day}_s{seed}"
    return label, out_dir, split, split_tag, day, seed


def _map_genome_to_mm_params(x):
    x = np.asarray(x, dtype=float)
    x = np.array([_clamp(v, lo, hi) for v,(lo,hi) in zip(x, BOUNDS)])

    # derive a participation ratio from gene 0
    pov = _clamp(0.005 + x[0] / 100.0, 0.01, 0.50)  # 0.01–0.50

    size = int(round(x[1]))
    skew_beta = float(x[2])
    level_spacing = float(x[3])
    spread_alpha = float(x[4])
    cancel_delay = int(round(x[5]))
    risk_aversion = float(x[6])
    inv_multiplier = float(x[7])

    num_ticks = max(2, min(60, int(round(size / 40))))
    window_size = max(2, min(200, int(round(level_spacing * 8))))
    wake_freq_s = max(1, min(60, int(round(2 + (10 - min(level_spacing, 10)) * pov * 5))))
    wake_up_freq = f"{wake_freq_s}S"
    backstop_qty = int(max(size, size * min(4, pov * 20)))
    max_inventory = int(max(backstop_qty * inv_multiplier, size * inv_multiplier * 2))

    # Send a broad set of synonyms so the agent accepts *something*.
    mm = {
        # participation
        "pov": pov,
        "participation_rate": pov,
        "mm_pov": pov,

        # size knobs (many forks choose one of these)
        "min_order_size": size,
        "order_size": size,
        "quote_size": size,
        "mm_min_order_size": size,

        # inventory aversion / risk
        "inventory_risk_aversion": risk_aversion,
        "inv_aversion": risk_aversion,
        "risk_aversion": risk_aversion,

        # skew/intensity
        "skew_gain": skew_beta,
        "skew": skew_beta,
        "skew_beta": skew_beta,
        "mm_skew_beta": skew_beta,

        # ladder geometry
        "level_spacing": level_spacing,
        "mm_level_spacing": level_spacing,
        "window_size": window_size,
        "mm_window_size": window_size,
        "num_ticks": num_ticks,
        "mm_num_ticks": num_ticks,

        # spread smoothing
        "spread_alpha": spread_alpha,
        "mm_spread_alpha": spread_alpha,

        # cancellation / pacing
        "cancel_delay": cancel_delay,
        "cancel_limit_delay": cancel_delay,
        "mm_cancel_limit_delay": cancel_delay,
        "wake_up_freq": wake_up_freq,
        "mm_wake_up_freq": wake_up_freq,

        # book support
        "backstop_quantity": backstop_qty,
        "mm_backstop_quantity": backstop_qty,
        "max_inventory": max_inventory,
        "inventory_limit": max_inventory,
        "mm_inventory_limit": max_inventory,
    }
    info = {
        "pov": pov,
        "size": size,
        "skew_beta": skew_beta,
        "level_spacing": level_spacing,
        "spread_alpha": spread_alpha,
        "cancel_delay": cancel_delay,
        "risk_aversion": risk_aversion,
        "inventory_multiplier": inv_multiplier,
        "window_size": window_size,
        "num_ticks": num_ticks,
        "wake_up_freq": wake_up_freq,
        "max_inventory": max_inventory,
    }
    return mm, info


def _resolve_lob_files(primary_file, directory):
    files = []
    if directory:
        dir_path = Path(str(directory)).expanduser()
        if dir_path.is_dir():
            files.extend(sorted(p for p in dir_path.glob("*.txt") if p.is_file()))
    if primary_file:
        p = Path(str(primary_file)).expanduser()
        if p.is_file():
            if p not in files:
                files.insert(0, p)
    return files


def _select_lob_path(args, eval_uid, combo_idx, dataset_override=None):
    if dataset_override:
        dataset_path = Path(dataset_override).expanduser().resolve()
        if dataset_path.is_file():
            return dataset_path, dataset_path.name
    files = getattr(args, "lob_files", None)
    if not files:
        return None, "synthetic"
    if len(files) == 1:
        path = files[0]
    else:
        span_default = len(getattr(args, "days", [])) * len(
            getattr(args, "seeds", [])
        )
        combos = max(1, int(getattr(args, "combo_span", span_default)))
        span = eval_uid * combos + combo_idx
        if getattr(args, "lob_random", False):
            seed = (args.cma_seed * 73477 + span * 104729) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
            idx = int(rng.randint(len(files)))
        else:
            idx = int(span % len(files))
        path = files[idx]
    return path, path.name

# ---------- parsing helpers ----------
NUM = r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?"
RE_HEADER = re.compile(r"^\s*Mean ending value by agent type\s*:\s*$", re.IGNORECASE)
RE_AGENT_LINE = re.compile(rf"^\s*(.+?)\s*:\s*({NUM})\s*$")
RE_PNL_INLINE = re.compile(
    rf"^\s*(?:Adaptive.*MarketMaker|ADAPTIVE[_ ]POV[_ ]MARKET[_ ]MAKER(?:[_ ]AGENT)?)\s*:\s*({NUM})\s*$",
    re.IGNORECASE
)
RE_MM_HOLDINGS = re.compile(
    r"^Final holdings for .*MARKET.*MAKER.*\{\s*ABM:\s*(-?\d+)\b.*?Marked to market:\s*(" + NUM + r")\s*$",
    re.IGNORECASE
)
def _num(s): return float(str(s).replace(",", ""))

def _parse_pnl_from_summary(so: str):
    pnl = {}
    lines = so.splitlines()
    in_block = False
    for ln in lines:
        if not in_block:
            if RE_HEADER.search(ln):
                in_block = True
            continue
        m = RE_AGENT_LINE.match(ln)
        if m:
            agent, val = m.group(1).strip(), _num(m.group(2))
            pnl[agent] = val
        else:
            if pnl:
                break
    return pnl or None

def _parse_pnl_and_inv(so: str, start_cash: float):
    # direct shortcut
    m = RE_PNL_INLINE.search(so)
    pnl_val = None
    if m:
        pnl_val = _num(m.group(1))

    # summary block
    if pnl_val is None:
        block = _parse_pnl_from_summary(so)
        if block:
            for k, v in block.items():
                ku = k.upper().replace(" ", "_")
                if "MARKET" in ku and "MAKER" in ku:
                    pnl_val = float(v); break

    # fallback to holdings
    inv_abs = None
    for ln in so.splitlines()[::-1]:
        m2 = RE_MM_HOLDINGS.search(ln)
        if m2:
            try:
                inv_abs = abs(float(m2.group(1)))
            except Exception:
                inv_abs = None
            try:
                m2m = _num(m2.group(2))
                if pnl_val is None:
                    pnl_val = float(m2m - start_cash)
            except Exception:
                pass
            break

    return pnl_val, inv_abs

# ---------- streaming runner ----------
def _reader_thread(pipe, acc):
    try:
        for line in iter(pipe.readline, ''):
            acc.append(line)
    except Exception:
        pass
    finally:
        try: pipe.close()
        except Exception: pass

def _run_abides(label, seed, day, cfg_path: Path, timeout_s: int):
    cmd = [
        sys.executable, "-u", ABIDES_PY,
        # custom wrapper lives in mm_configs/rmsc03.py
        "-c", "mm_configs.rmsc03",
        "-t", "ABM",
        "-d", day,
        "-s", str(seed),
        "-l", label,
    ]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([
        str(ROOT),
        str(ABIDES_CORE),
        *(p for p in [existing] if p),
    ])
    env["MM_MVP_CFG"] = str(cfg_path)
    env["ABIDES_DISABLE_BOOKLOG"] = "1"
    env["MM_DISABLE_BOOKLOG"] = "1"

    t0 = time.time()
    killed = False
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1
    )
    so_lines, se_lines = [], []
    t_so = threading.Thread(target=_reader_thread, args=(proc.stdout, so_lines), daemon=True)
    t_se = threading.Thread(target=_reader_thread, args=(proc.stderr, se_lines), daemon=True)
    t_so.start(); t_se.start()

    while True:
        rc = proc.poll()
        if rc is not None:
            break
        if (time.time() - t0) > timeout_s:
            killed = True
            try: proc.terminate()
            except Exception: pass
            for _ in range(30):
                time.sleep(0.1)
                if proc.poll() is not None:
                    break
            if proc.poll() is None:
                try: proc.kill()
                except Exception: pass
            break
        time.sleep(0.1)

    t_so.join(timeout=1.0)
    t_se.join(timeout=1.0)

    dur = time.time() - t0
    rc = proc.returncode if proc.returncode is not None else 124
    tag = "timeout" if killed or rc == 124 else "ok"
    so = "".join(so_lines)
    se = "".join(se_lines)
    return rc, dur, so, se, tag

# ---------- per-eval ----------
def _write_cfg(out_dir: Path, payload: dict):
    p = out_dir / "mvp_cfg.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return p

def _tail(text: str, n=40):
    lines = (text or "").splitlines()
    return "\n".join(lines[-n:]) if lines else ""

def _evaluate_once(genome, gen, idx, combo, args, combo_idx, base_dir=None, event_bus=None, event_queue=None):
    """Run a single ABIDES evaluation for the provided genome and scenario."""
    label, out_dir, split, split_tag, day, seed = _make_eval_paths(gen, idx, combo, base_dir=base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mm_params, mm_info = _map_genome_to_mm_params(genome)
    mm_summary = (
        f"pov={mm_info['pov']:.3f} size={mm_info['size']} skew={mm_info['skew_beta']:.2f} "
        f"spacing={mm_info['level_spacing']:.2f} spread_a={mm_info['spread_alpha']:.2f} "
        f"cancel={mm_info['cancel_delay']}ns risk={mm_info['risk_aversion']:.2f} "
        f"inv_mult={mm_info['inventory_multiplier']:.2f} freq={mm_info['wake_up_freq']}"
    )

    pop_eff = getattr(args, "popsize_effective", args.popsize)
    eval_uid = (gen - 1) * int(pop_eff) + idx

    dataset_override = combo.get("dataset")
    dataset_path, dataset_tag = _select_lob_path(args, eval_uid, combo_idx, dataset_override=dataset_override)

    lob_cfg = None
    if dataset_path:
        dataset_resolved = dataset_path.expanduser().resolve()
        lob_cfg = {
            "file": str(dataset_resolved),
            "freq": args.lob_freq,
            "scale": args.lob_scale,
            "offset": args.lob_offset,
            "enforce_monotonic": not args.lob_no_monotonic,
        }
        dataset_tag = dataset_resolved.name
    else:
        dataset_tag = dataset_tag or "synthetic"

    payload = {
        "session_seconds": int(args.seconds),
        "session_minutes": int(args.minutes),
        "thin_agents": {
            "ValueAgent":     args.thin_value,
            "NoiseAgent":     args.thin_noise,
            "MomentumAgent":  args.thin_momentum,
            "ExecutionAgent": args.thin_execution,
        },
        "disable_booklog": True,
        "mm_params": mm_params,
    }
    if lob_cfg:
        payload["lob_dataset"] = lob_cfg
    cfg_path = _write_cfg(out_dir, payload)

    start_ts = time.time()
    dataset_hint = dataset_tag if dataset_tag and dataset_tag != "synthetic" else None
    target_queue = event_queue if event_queue is not None else EVENT_QUEUE
    if event_bus:
        event_bus.publish(
            EvaluationStartEvent(
                label=label,
                generation=int(gen),
                individual_index=int(idx),
                split=split,
                day=str(day),
                seed=int(seed),
                dataset_hint=dataset_hint,
                timestamp=start_ts,
            )
        )
    if target_queue is not None:
        target_queue.put({
            "type": "start",
            "label": label,
            "generation": int(gen),
            "individual_index": int(idx),
            "split": split,
            "day": str(day),
            "seed": int(seed),
            "dataset": dataset_hint or dataset_tag,
            "timestamp": start_ts,
        })

    py_path_entries = [str(ROOT), str(ABIDES_CORE)]
    if existing := os.environ.get("PYTHONPATH"):
        py_path_entries.append(existing)
    with open(out_dir / "cmdline.txt", "w", encoding="utf-8") as f:
        f.write(
            "CMD: {} -u {} -c {} -t ABM -d {} -s {} -l {}\n".format(
                sys.executable,
                ABIDES_PY,
                "mm_configs.rmsc03",
                day,
                seed,
                label,
            )
        )
        f.write(f"PYTHONPATH={'{}'}\nMM_MVP_CFG={cfg_path}\n".format(os.pathsep.join(py_path_entries)))

    rc, dur, so, se, tag = _run_abides(label, seed, day, cfg_path, timeout_s=args.timeout)

    (out_dir / "cmd_stdout.txt").write_text(so or "", encoding="utf-8", errors="ignore")
    (out_dir / "cmd_stderr.txt").write_text(se or "", encoding="utf-8", errors="ignore")

    mm_summary_path = out_dir / "wrapper_seen.txt"
    if not mm_summary_path.exists():
        alt = ROOT / "wrapper_seen.txt"
        mm_summary_path = alt if alt.exists() else mm_summary_path
    mm_marker = mm_summary_path.exists()
    mm_attrs_applied = []
    dataset_seen = None
    if mm_marker:
        try:
            lines = mm_summary_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if lines:
                summary = json.loads(lines[-1])
                mm_attrs_applied = summary.get("mm_attrs_applied") or []
                dataset_seen = summary.get("lob_dataset")
        except Exception:
            mm_attrs_applied = []
            dataset_seen = None

    if dataset_seen:
        if isinstance(dataset_seen, dict) and dataset_seen.get("file"):
            dataset_tag = Path(dataset_seen["file"]).name
        else:
            dataset_tag = str(dataset_seen)

    pnl, inv_abs = _parse_pnl_and_inv(so or "", args.mm_start_cash)
    inv_abs = inv_abs if inv_abs is not None else 0.0

    reason = tag
    if rc != 0 and tag != "timeout":
        reason = f"rc={rc}"
    if not mm_marker:
        reason = "wrapper_not_seen"
    if pnl is None and reason == "ok":
        reason = "pnl_not_found"

    session_seconds = args.seconds if args.seconds > 0 else max(1, int(args.minutes) * 60)
    pnl_normalized = float(pnl) / args.mm_start_cash if (pnl is not None and args.mm_start_cash) else None
    pnl_per_second = (pnl_normalized / session_seconds) if (pnl_normalized is not None and session_seconds > 0) else None
    log_return = None
    if pnl_normalized is not None and (1.0 + pnl_normalized) > 0:
        log_return = math.log1p(pnl_normalized)
    inventory_normalized = float(inv_abs) / args.mm_start_cash if args.mm_start_cash else float(inv_abs)

    drawdown_penalty = 0.0
    if args.score_mode == "normalized" and pnl_per_second is not None:
        shortfall = max(0.0, -pnl_per_second - args.drawdown_threshold)
        drawdown_penalty = args.drawdown_penalty * shortfall
    elif args.score_mode == "raw" and pnl is not None and args.mm_start_cash:
        normalized_loss = -pnl / args.mm_start_cash
        shortfall_raw = max(0.0, normalized_loss - args.drawdown_threshold)
        drawdown_penalty = args.drawdown_penalty * shortfall_raw * args.mm_start_cash

    if (rc != 0) or (not mm_marker) or (pnl is None):
        score = -1_000_000.0
        score_detail = {"mode": args.score_mode, "reason": "invalid_run"}
    else:
        if args.score_mode == "normalized":
            penalty = args.inv_penalty * inventory_normalized
            base = log_return if log_return is not None else (pnl_per_second if pnl_per_second is not None else float(pnl))
            penalty += drawdown_penalty
        else:
            penalty = args.inv_penalty * float(inv_abs)
            base = log_return if log_return is not None else float(pnl)
            penalty += drawdown_penalty
        loss_penalty = 0.0
        if pnl is not None:
            if args.mm_start_cash:
                loss_penalty = max(0.0, -float(pnl) / args.mm_start_cash)
            else:
                loss_penalty = max(0.0, -float(pnl))
            loss_penalty *= float(getattr(args, "loss_penalty", cfg.DEFAULT_LOSS_PENALTY))
        score = float(base) - float(penalty) - float(loss_penalty)
        clip_limit = getattr(args, "drawdown_clip", None)
        if base is None:
            score = -1_000_000.0
        score_before_clip = score
        if clip_limit:
            score = max(score, -float(clip_limit))
        score_detail = {
            "mode": args.score_mode,
            "base": base,
            "penalty": penalty,
            "session_seconds": session_seconds,
            "inventory_normalized": inventory_normalized,
            "pnl_per_second": pnl_per_second,
            "log_return": log_return,
            "drawdown_penalty": drawdown_penalty,
            "clip_limit": clip_limit,
            "loss_penalty": loss_penalty,
            "clipped": score != score_before_clip,
        }

    # Helpful debug: show the parsed numbers & a short tail when failing.
    if score <= -999999.9:
        (out_dir / "stdout_tail.txt").write_text(_tail(so, 40), encoding="utf-8", errors="ignore")
        tail_note = " tail=stdout_tail.txt"
    else:
        tail_note = ""

    so_b = len((so or "").encode("utf-8"))
    se_b = len((se or "").encode("utf-8"))
    if mm_attrs_applied:
        applied_parts = []
        for entry in mm_attrs_applied:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                applied_parts.append(f"{entry[0]}={entry[1]}")
            else:
                applied_parts.append(str(entry))
        applied_note = f" applied={'|'.join(applied_parts)}"
    else:
        applied_note = ""

    if target_queue is None:
        print(
            f"[eval g{gen:03d} i{idx:02d} split={split} d{day} s{seed}] "
            f"rc={rc} dur={dur:.1f}s score={score:.1f} mm_marker={int(mm_marker)} "
            f"pnl={pnl if pnl is not None else 'NA'} inv_abs={inv_abs:.0f} "
            f"stdout={so_b}B stderr={se_b}B reason={reason}{tail_note} "
            f"params={mm_summary} data={dataset_tag}{applied_note}"
        )

    summary_payload = {
        "generation": int(gen),
        "pop_index": int(idx),
        "day": day,
        "seed": int(seed),
        "dataset": dataset_tag,
        "split": split,
        "score_mode": args.score_mode,
        "score": float(score),
        "pnl": float(pnl) if pnl is not None else None,
        "pnl_normalized": pnl_normalized,
        "pnl_per_second": pnl_per_second,
        "inventory_abs": float(inv_abs),
        "inventory_normalized": inventory_normalized,
        "duration_sec": float(dur),
        "rc": int(rc),
        "reason": reason,
        "score_components": score_detail,
        "genome": [float(v) for v in np.asarray(genome, dtype=float)],
        "mm_params": mm_info,
        "mm_summary": mm_summary,
        "mm_applied": [(str(a), float(v) if isinstance(v, (int, float)) else v) for a, v in mm_attrs_applied] if mm_attrs_applied else [],
        "lob_dataset": lob_cfg,
    }
    try:
        (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[warn] failed to write summary.json: {exc}", file=sys.stderr)

    end_ts = time.time()
    result = EvaluationResult(
        label=label,
        generation=int(gen),
        individual_index=int(idx),
        split=split,
        day=str(day),
        seed=int(seed),
        dataset=dataset_tag,
        score=float(score),
        pnl=float(pnl) if pnl is not None else None,
        inventory_abs=float(inv_abs),
        reason=reason,
        duration=float(dur),
        start_time=start_ts,
        end_time=end_ts,
        mm_summary=mm_summary,
        genome_vector=[float(v) for v in np.asarray(genome, dtype=float)],
        score_components=score_detail,
    )

    if event_bus:
        event_bus.publish(EvaluationCompleteEvent(result))
    if target_queue is not None:
        target_queue.put({
            "type": "complete",
            "label": label,
            "generation": int(gen),
            "individual_index": int(idx),
            "split": split,
            "day": str(day),
            "seed": int(seed),
            "dataset": dataset_tag,
            "score": float(score),
            "pnl": float(pnl) if pnl is not None else None,
            "inventory_abs": float(inv_abs),
            "duration": float(dur),
            "reason": reason,
            "timestamp": end_ts,
        })

    return result

def _evaluate_mean(genome, gen, idx, args, event_bus=None, event_queue=None):
    """Return the aggregated training score for one CMA-ES individual."""
    target_queue = event_queue if event_queue is not None else EVENT_QUEUE
    train_scores = []
    val_scores = []
    other_scores = []
    all_scores = []
    combo_idx = 0
    for combo in getattr(args, "current_combo_plan", []):
        result = _evaluate_once(
            genome,
            gen,
            idx,
            combo,
            args,
            combo_idx,
            event_bus=event_bus,
            event_queue=target_queue,
        )
        score = result.score
        split = combo.get("split", "train").lower()
        all_scores.append(score)
        if split.startswith("train"):
            train_scores.append(score)
        elif split.startswith("val"):
            val_scores.append(score)
        else:
            other_scores.append(score)
        combo_idx += 1
    robust_quantile = float(getattr(args, "robust_quantile", DEFAULT_ROBUST_QUANTILE))
    robust_quantile = min(max(robust_quantile, 0.0), 1.0)
    trim_frac = float(getattr(args, "score_trim", 0.0))

    if train_scores:
        train_cvar = _cvar(train_scores, robust_quantile, trim_frac)
    else:
        train_cvar = None

    if train_cvar is None:
        if all_scores:
            return float(np.mean(all_scores))
        return -1_000_000.0

    score_result = float(train_cvar)

    if val_scores and getattr(args, "validation_weight", 0.0) != 0.0:
        val_mean = float(np.mean(val_scores))
        score_result += float(args.validation_weight) * val_mean

    return score_result

# ---------- CMA-ES loop ----------
def run_cmaes(args, event_bus: Optional[EventBus] = None, event_queue=None, mp_context=None):
    """Optimise market-maker parameters with CMA-ES."""
    global EVENT_QUEUE
    EVENT_QUEUE = event_queue
    silence_output = event_queue is not None
    x0 = MID.copy()
    sigma0 = SIGMA0

    args.score_mode = (args.score_mode or "normalized").lower()
    if args.score_mode not in {"normalized", "raw"}:
        raise ValueError("--score-mode must be 'normalized' or 'raw'")

    shutil.rmtree(EVALS, ignore_errors=True)
    EVALS.mkdir(parents=True, exist_ok=True)
    best_hist_path = RUNS / "best_history.json"
    if best_hist_path.exists():
        best_hist_path.unlink()
    if not silence_output:
        print("[mm_cmaes] Cleared previous evaluation outputs.")

    if not getattr(args, "lob_files", None):
        args.lob_files = _resolve_lob_files(args.lob_file, args.lob_dir)

    if getattr(args, "lob_random", False) and not args.lob_files:
        print("[warn] --lob-random requested but no dataset files were discovered.", file=sys.stderr)

    smoke = getattr(args, "smoke", False)
    if smoke:
        args.quick = True

    if args.quick:
        popsize = 4
        gens = 2
        args.train_days = args.train_days[:1] if args.train_days else list(DEFAULT_TRAIN_DAYS[:1])
        args.train_seeds = args.train_seeds[:1] if args.train_seeds else list(DEFAULT_TRAIN_SEEDS[:1])
        args.val_days = args.val_days[:1] if args.val_days else list(args.train_days)
        args.val_seeds = args.val_seeds[:1] if args.val_seeds else list(args.train_seeds)
        args.train_dataset_paths = args.train_dataset_paths[:1] if args.train_dataset_paths else args.train_dataset_paths
        args.val_dataset_paths = args.val_dataset_paths[:1] if args.val_dataset_paths else args.val_dataset_paths
        args.warmup_train_dataset_paths = args.warmup_train_dataset_paths[:1] if args.warmup_train_dataset_paths else args.warmup_train_dataset_paths
        args.warmup_val_dataset_paths = args.warmup_val_dataset_paths[:1] if args.warmup_val_dataset_paths else args.warmup_val_dataset_paths
        if args.seconds <= 0:
            args.seconds = DEFAULT_SECONDS
    else:
        popsize = args.popsize
        gens = args.gens

    if smoke:
        popsize = min(popsize, 2)
        gens = min(gens, 1)
        args.train_days = args.train_days[:1] if args.train_days else list(DEFAULT_TRAIN_DAYS[:1])
        args.train_seeds = args.train_seeds[:1] if args.train_seeds else list(DEFAULT_TRAIN_SEEDS[:1])
        args.val_days = args.val_days[:1] if args.val_days else list(args.train_days)
        args.val_seeds = args.val_seeds[:1] if args.val_seeds else list(args.train_seeds)
        args.train_dataset_paths = args.train_dataset_paths[:1] if args.train_dataset_paths else args.train_dataset_paths
        args.val_dataset_paths = args.val_dataset_paths[:1] if args.val_dataset_paths else args.val_dataset_paths
        args.warmup_train_dataset_paths = args.warmup_train_dataset_paths[:1] if args.warmup_train_dataset_paths else args.warmup_train_dataset_paths
        args.warmup_val_dataset_paths = args.warmup_val_dataset_paths[:1] if args.warmup_val_dataset_paths else args.warmup_val_dataset_paths
        if args.seconds <= 0:
            args.seconds = 5
        else:
            args.seconds = min(args.seconds, 5)
        args.minutes = min(args.minutes, 1)

    dataset_union = list(dict.fromkeys(
        (args.train_dataset_paths or [])
        + (args.val_dataset_paths or [])
        + (args.test_dataset_paths or [])
    ))
    if dataset_union:
        args.lob_files = dataset_union

    warmup_train_ds = args.warmup_train_dataset_paths if args.warmup_train_dataset_paths else args.train_dataset_paths
    warmup_val_ds = args.warmup_val_dataset_paths if args.warmup_val_dataset_paths else args.val_dataset_paths
    train_ds_full = args.train_dataset_paths
    val_ds_full = args.val_dataset_paths

    warmup_combo_plan = prepare_combo_plan(
        args.train_days,
        args.train_seeds,
        warmup_train_ds,
        args.val_days,
        args.val_seeds,
        warmup_val_ds,
    )
    warmup_combo_count = len(warmup_combo_plan)

    args.current_combo_plan = warmup_combo_plan
    args.combo_span = warmup_combo_count

    if args.quick or smoke:
        args.max_evals = min(args.max_evals, popsize * gens * warmup_combo_count)

    eval_budget = args.max_evals if getattr(args, "max_evals", None) else None
    if eval_budget and eval_budget > 0:
        max_gens = max(1, math.ceil(eval_budget / popsize))
        gens = min(gens, max_gens)

    args.popsize_effective = int(popsize)

    run_start = time.time()
    silence_output = (event_queue is not None or EVENT_QUEUE is not None)

    es = cma.CMAEvolutionStrategy(x0, sigma0, {"popsize": int(popsize), "seed": args.cma_seed})

    evals_done = 0
    best_score_so_far = float("-inf")
    stagnant_gens = 0
    patience = max(0, getattr(args, "early_stop_patience", 0))
    workers = max(1, min(int(getattr(args, "workers", 1)), popsize))
    if event_bus is not None and workers > 1:
        print("[mm_cmaes] GUI active – forcing workers=1 for live updates.")
        workers = 1
    args.workers = workers
    best_history = []

    target_queue_meta = event_queue if event_queue is not None else EVENT_QUEUE
    executor = None
    if workers > 1:
        ctx = mp_context
        if ctx is None:
            ctx = multiprocessing.get_context()
        init = _worker_init if target_queue_meta is not None else None
        initargs = (target_queue_meta,) if target_queue_meta is not None else ()
        pool_kwargs = {"max_workers": workers}
        if ctx is not None:
            pool_kwargs["mp_context"] = ctx
        if init is not None:
            pool_kwargs["initializer"] = init
            pool_kwargs["initargs"] = initargs
        executor = ProcessPoolExecutor(**pool_kwargs)

    if target_queue_meta is not None:
        target_queue_meta.put({
            "type": "meta",
            "total_generations": int(gens),
            "workers": int(workers),
        })

    try:
        for gen in range(1, gens + 1):
            if args.warmup_generations and gen <= args.warmup_generations:
                train_ds_current = warmup_train_ds
                val_ds_current = warmup_val_ds
                args.dataset_agg_current = getattr(args, "warmup_dataset_agg", args.dataset_agg)
            else:
                train_ds_current = train_ds_full
                val_ds_current = val_ds_full
                if getattr(args, "dataset_agg_switch_gen", 0) and gen > args.dataset_agg_switch_gen:
                    args.dataset_agg_current = args.dataset_agg_post
                else:
                    args.dataset_agg_current = args.dataset_agg

            current_combos = prepare_combo_plan(
                args.train_days,
                args.train_seeds,
                train_ds_current,
                args.val_days,
                args.val_seeds,
                val_ds_current,
            )
            args.current_combo_plan = current_combos
            args.combo_span = len(current_combos)

            X = es.ask()
            population = list(enumerate(X))
            if eval_budget:
                remaining = max(0, eval_budget - evals_done)
                if remaining <= 0:
                    if not silence_output:
                        print("Evaluation budget exhausted before generation start.")
                    break
                population = population[:remaining]

            if not population:
                break

            F_map = {}
            total = len(population)
            show_progress = (not getattr(args, "no_progress", False)) and sys.stdout.isatty()
            completed = 0
            color_prefix = ANSI_BLUE if _use_color(args) else ""
            color_reset = ANSI_RESET if _use_color(args) else ""

            def _print_progress():
                if not show_progress:
                    return
                bar = _progress_bar(completed, total)
                sys.stdout.write(f"\r{color_prefix}[Gen {gen}/{gens}] {bar}{color_reset}")
                sys.stdout.flush()

            if show_progress:
                _print_progress()

            if executor is None:
                for i, x in population:
                    score = _evaluate_mean(
                        x,
                        gen,
                        i,
                        args,
                        event_bus=event_bus,
                        event_queue=event_queue,
                    )
                    F_map[i] = -score
                    evals_done += 1
                    completed += 1
                    _print_progress()
            else:
                futures = {
                    executor.submit(_evaluate_mean, X[i], gen, i, args, None, None): i
                    for i, _ in population
                }
                for future in as_completed(futures):
                    i = futures[future]
                    score = future.result()
                    F_map[i] = -score
                    evals_done += 1
                    completed += 1
                    _print_progress()

            evaluated_indices = [i for i, _ in population if i in F_map]
            if not evaluated_indices:
                if not silence_output:
                    print("No evaluations completed for this generation.")
                break

            if show_progress:
                sys.stdout.write("\n")

            evaluated_candidates = [X[i] for i in evaluated_indices]
            F = [F_map[i] for i in evaluated_indices]

            es.tell(evaluated_candidates, F)

            gen_scores = [-val for val in F]
            gen_best_score = max(gen_scores)
            gen_mean_score = float(np.mean(gen_scores))
            best_idx_local = evaluated_indices[int(np.argmax(gen_scores))]
            best = X[best_idx_local]
            best_str = _fmt_score(gen_best_score, args)
            mean_str = _fmt_score(gen_mean_score, args)
            if not silence_output:
                print(
                    f"[gen {gen}/{gens}] best={best_str} mean={mean_str} "
                    f"genome={np.array2string(best, precision=6)} evals={evals_done}"
                )

            best_history.append({
                "generation": gen,
                "best_score": gen_best_score,
                "mean_score": gen_mean_score,
                "genome": best.tolist() if isinstance(best, np.ndarray) else list(best),
            })

            if event_bus:
                event_bus.publish(
                    GenerationSummaryEvent(
                        generation=int(gen),
                        best_score=float(gen_best_score),
                        mean_score=float(gen_mean_score),
                        evaluations_completed=int(evals_done),
                        timestamp=time.time(),
                    )
                )
            target_queue_gen = event_queue if event_queue is not None else EVENT_QUEUE
            if target_queue_gen is not None:
                target_queue_gen.put({
                    "type": "generation",
                    "generation": int(gen),
                    "best_score": float(gen_best_score),
                    "mean_score": float(gen_mean_score),
                    "evaluations_completed": int(evals_done),
                    "timestamp": time.time(),
                })

            if gen_best_score > best_score_so_far + 1e-9:
                best_score_so_far = gen_best_score
                stagnant_gens = 0
            else:
                stagnant_gens += 1
                if patience and stagnant_gens >= patience:
                    if not silence_output:
                        print(f"Early stopping after {gen} generations (patience={patience}).")
                    break

            if eval_budget and evals_done >= eval_budget:
                if not silence_output:
                    print("Evaluation budget exhausted; stopping CMA-ES loop.")
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    try:
        (RUNS / "best_history.json").write_text(json.dumps(best_history, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[warn] failed to write best_history.json: {exc}", file=sys.stderr)

    if not silence_output:
        print(f"Done. Evaluations completed: {evals_done}. Per-eval files in {EVALS}.")

    if not getattr(args, "no_post_analysis", False):
        original_quiet = getattr(args, "quiet", False)
        args.quiet = silence_output
        try:
            run_analysis(args)
        except Exception as exc:
            print(f"[warn] post-run analysis failed: {exc}", file=sys.stderr)
        finally:
            args.quiet = original_quiet

    best_genome = _load_best_genome()
    if best_genome is not None and not getattr(args, "skip_holdout", False):
        holdout_root = getattr(args, "holdout_output", None)
        base_dir = Path(holdout_root) if holdout_root else None
        if args.val_dataset_paths:
            val_output = (base_dir / "validation") if base_dir else None
            run_holdout(
                args,
                best_genome,
                "validation",
                args.val_days,
                args.val_seeds,
                args.val_dataset_paths,
                output_dir=val_output,
            )
        if args.test_dataset_paths and args.test_days and args.test_seeds:
            test_output = (base_dir / "test") if base_dir else None
            run_holdout(
                args,
                best_genome,
                "test",
                args.test_days,
                args.test_seeds,
                args.test_dataset_paths,
                output_dir=test_output,
            )

    if event_bus:
        event_bus.publish(
            RunCompleteEvent(
                evaluations_completed=int(evals_done),
                duration=float(time.time() - run_start),
                timestamp=time.time(),
            )
        )
    target_queue_final = event_queue if event_queue is not None else EVENT_QUEUE
    if target_queue_final is not None:
        target_queue_final.put({
            "type": "run_complete",
            "evaluations_completed": int(evals_done),
            "duration": float(time.time() - run_start),
            "timestamp": time.time(),
        })
    EVENT_QUEUE = None


def _iter_eval_summaries():
    if not EVALS.exists():
        return
    for path in sorted(EVALS.rglob("summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] failed to parse {path}: {exc}", file=sys.stderr)
            continue
        data["_summary_path"] = str(path)
        yield data


def _format_float(val, precision=3):
    if val is None or isinstance(val, str) and val.upper() == "NAN":
        return "NA"
    try:
        return f"{float(val):.{precision}f}"
    except Exception:
        return str(val)


def _summary_stats(values):
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return None
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci = 1.96 * std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    median = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return {
        "count": int(arr.size),
        "mean": mean,
        "std": std,
        "ci95": ci,
        "median": median,
        "q25": q25,
        "q75": q75,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _load_genome_spec(spec):
    path = Path(spec)
    payload = None
    if path.exists() and path.is_file():
        text = path.read_text(encoding="utf-8", errors="ignore")
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                if "genome" in data:
                    payload = data["genome"]
                elif "vector" in data:
                    payload = data["vector"]
            elif isinstance(data, list):
                payload = data
            if payload is None:
                # fallback: treat file content as comma/space separated numbers
                payload = re.split(r"[\s,]+", text.strip())
        except json.JSONDecodeError:
            payload = re.split(r"[\s,]+", text.strip())
    else:
        payload = re.split(r"[\s,]+", spec.strip())

    try:
        genome = np.array([float(x) for x in payload if str(x).strip()], dtype=float)
    except Exception:
        return None
    if genome.ndim != 1 or genome.size != len(BOUNDS):
        return None
    return genome


def _load_best_genome(history_path=None):
    """Fetch the latest genome recorded in best_history.json."""
    path = Path(history_path) if history_path else (RUNS / "best_history.json")
    if not path.exists():
        return None
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not history:
        return None
    genome = history[-1].get("genome")
    if genome is None:
        return None
    try:
        return np.array(genome, dtype=float)
    except Exception:
        return None


def run_holdout(args, genome, split_name, days, seeds, datasets, output_dir=None):
    """Replay a genome on the requested split and write a summary report."""
    if genome is None:
        print(f"[{split_name}] No genome provided for hold-out run.", file=sys.stderr)
        return

    combos = make_split_combos(split_name, days, seeds, datasets)
    if not combos:
        print(f"[{split_name}] No scenarios configured; skipping hold-out run.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(output_dir) if output_dir else RUNS / f"{split_name}_holdout" / timestamp
    base.mkdir(parents=True, exist_ok=True)

    prev_plan = getattr(args, "current_combo_plan", None)
    prev_span = getattr(args, "combo_span", None)
    args.current_combo_plan = combos
    args.combo_span = len(combos)

    combo_idx = 0
    for local_idx, combo in enumerate(combos):
        _evaluate_once(genome, 0, local_idx, combo, args, combo_idx, base_dir=base, event_bus=None)
        combo_idx += 1

    summaries = []
    for path in sorted(base.rglob("summary.json")):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"[{split_name}] failed to parse {path}: {exc}", file=sys.stderr)

    args.current_combo_plan = prev_plan
    args.combo_span = prev_span

    if not summaries:
        print(f"[{split_name}] No summaries produced.")
        return

    score_stats = _summary_stats([s.get("score") for s in summaries])
    pnl_stats = _summary_stats([s.get("pnl") for s in summaries])
    inv_stats = _summary_stats([s.get("inventory_abs") for s in summaries])

    by_dataset = {}
    for summary in summaries:
        dataset = summary.get("dataset", "NA")
        bucket = by_dataset.setdefault(dataset, {"scores": [], "pnls": [], "inventory": []})
        if summary.get("score") is not None:
            bucket["scores"].append(summary["score"])
        if summary.get("pnl") is not None:
            bucket["pnls"].append(summary["pnl"])
        if summary.get("inventory_abs") is not None:
            bucket["inventory"].append(summary["inventory_abs"])

    lines = []
    lines.append(f"# {split_name.title()} Hold-out Report")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Genome: {genome.tolist()}")
    lines.append(f"Scenarios evaluated: {len(summaries)}")
    lines.append("")
    lines.append("## Aggregate Statistics")
    lines.append("|metric|mean+/-ci|median|min|max|")
    lines.append("|------|---------|------|---|---|")
    if score_stats:
        lines.append(
            f"|Score|{_format_float(score_stats['mean'],3)}+/-{_format_float(score_stats['ci95'],3)}|{_format_float(score_stats['median'],3)}|{_format_float(score_stats['min'],3)}|{_format_float(score_stats['max'],3)}|"
        )
    if pnl_stats:
        lines.append(
            f"|PnL|{_format_float(pnl_stats['mean'],1)}+/-{_format_float(pnl_stats['ci95'],1)}|{_format_float(pnl_stats['median'],1)}|{_format_float(pnl_stats['min'],1)}|{_format_float(pnl_stats['max'],1)}|"
        )
    if inv_stats:
        lines.append(
            f"|Inventory|{_format_float(inv_stats['mean'],0)}+/-{_format_float(inv_stats['ci95'],0)}|{_format_float(inv_stats['median'],0)}|{_format_float(inv_stats['min'],0)}|{_format_float(inv_stats['max'],0)}|"
        )

    lines.append("")
    lines.append("## Dataset Breakdown")
    lines.append("|dataset|runs|score_mean+/-ci|pnl_mean+/-ci|inv_mean|")
    lines.append("|-------|----|----------------|---------------|--------|")
    for dataset, metrics in sorted(by_dataset.items()):
        s_stats = _summary_stats(metrics["scores"])
        p_stats = _summary_stats(metrics["pnls"])
        i_stats = _summary_stats(metrics["inventory"])
        lines.append(
            "|{dataset}|{runs}|{score}|{pnl}|{inv}|".format(
                dataset=dataset,
                runs=len(metrics["scores"]) or len(metrics["pnls"]),
                score=f"{_format_float(s_stats['mean'],3)}+/-{_format_float(s_stats['ci95'],3)}" if s_stats else "NA",
                pnl=f"{_format_float(p_stats['mean'],1)}+/-{_format_float(p_stats['ci95'],1)}" if p_stats else "NA",
                inv=_format_float(i_stats['mean'],0) if i_stats else "NA",
            )
        )

    report_path = base / f"{split_name}_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[{split_name}] Completed hold-out evaluation for {len(summaries)} scenarios. Report: {report_path}")

def run_analysis(args):
    quiet = getattr(args, "quiet", False)
    summaries = list(_iter_eval_summaries())
    if not summaries:
        if not quiet:
            print(f"[analysis] No evaluation summaries found under {EVALS}.")
        return

    total = len(summaries)
    top_k = max(1, int(getattr(args, "analysis_top_k", 10)))
    summaries_sorted = sorted(summaries, key=lambda d: d.get("score", float("-inf")), reverse=True)
    top = summaries_sorted[:top_k]
    best = top[0]

    best_gen = int(best.get("generation", -1))
    best_idx = int(best.get("pop_index", -1))
    same_genome = [s for s in summaries if int(s.get("generation", -1)) == best_gen and int(s.get("pop_index", -1)) == best_idx]

    dataset_top_counter = Counter(s.get("dataset") for s in top if s.get("dataset"))
    dataset_all_counter = Counter(s.get("dataset") for s in summaries if s.get("dataset"))
    replicate_dataset_counter = Counter(s.get("dataset") for s in same_genome if s.get("dataset"))

    best_genome = best.get("genome") or []
    mm_params = best.get("mm_params") or {}
    score_components = best.get("score_components") or {}

    replicate_scores = [float(s.get("score")) for s in same_genome if s.get("score") is not None]
    replicate_pnls = [float(s.get("pnl")) for s in same_genome if s.get("pnl") is not None]

    split_metrics = {}
    for entry in summaries:
        split = entry.get("split", "train")
        bucket = split_metrics.setdefault(split, {"scores": [], "pnls": [], "inventory": []})
        if entry.get("score") is not None:
            bucket["scores"].append(float(entry["score"]))
        if entry.get("pnl") is not None:
            bucket["pnls"].append(float(entry["pnl"]))
        if entry.get("inventory_abs") is not None:
            bucket["inventory"].append(float(entry["inventory_abs"]))

    train_scores_all = [float(s.get("score")) for s in summaries if s.get("split", "").lower().startswith("train") and s.get("score") is not None]
    val_scores_all = [float(s.get("score")) for s in summaries if s.get("split", "").lower().startswith("val") and s.get("score") is not None]
    trim_frac = float(getattr(args, "score_trim", 0.0))
    cvar_alpha = float(getattr(args, "robust_quantile", DEFAULT_ROBUST_QUANTILE))
    train_cvar = _cvar(train_scores_all, cvar_alpha, trim_frac)
    train_mean = float(np.mean(train_scores_all)) if train_scores_all else None
    val_mean = float(np.mean(val_scores_all)) if val_scores_all else None

    lines = []
    lines.append("# CMA-ES Outlier Analysis")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Evaluations scanned: {total}")
    lines.append("")
    lines.append(f"Top {len(top)} evaluations by score (higher is better):")
    lines.append("|rank|gen|idx|seed|split|score|PnL|inventory|dataset|pov|risk|inv_mult|")
    lines.append("|---|---|---|---|-----|-----|----|---------|-------|---|----|--------|")
    for rank, entry in enumerate(top, 1):
        pov = entry.get("mm_params", {}).get("pov")
        risk = entry.get("mm_params", {}).get("risk_aversion")
        inv_mult = entry.get("mm_params", {}).get("inventory_multiplier")
        lines.append(
            "|{rank}|g{gen:03d}|i{idx:02d}|{seed}|{split}|{score}|{pnl}|{inv}|{dataset}|{pov}|{risk}|{inv_mult}|".format(
                rank=rank,
                gen=int(entry.get("generation", 0)),
                idx=int(entry.get("pop_index", 0)),
                seed=int(entry.get("seed", 0)),
                split=entry.get("split", "train"),
                score=_format_float(entry.get("score"), 3),
                pnl=_format_float(entry.get("pnl"), 1),
                inv=_format_float(entry.get("inventory_abs"), 0),
                dataset=entry.get("dataset", "NA"),
                pov=_format_float(pov, 3),
                risk=_format_float(risk, 3),
                inv_mult=_format_float(inv_mult, 2),
            )
        )

    lines.append("")
    lines.append("## Best Evaluation Genome")
    lines.append(f"Summary path: `{best.get('_summary_path')}`")
    lines.append("")
    lines.append("Genotype vector:")
    lines.append("```json")
    if best_genome:
        lines.append(json.dumps(best_genome, indent=2))
    else:
        lines.append("[")
        lines.append("  // not recorded for this evaluation")
        lines.append("]")
    lines.append("```")
    lines.append("")
    lines.append("Mapped market-maker parameters:")
    lines.append("```json")
    lines.append(json.dumps(mm_params, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("Score components:")
    lines.append("```json")
    lines.append(json.dumps(score_components, indent=2))
    lines.append("```")

    lines.append("")
    lines.append("### Replication Across Seeds/Days")
    lines.append(f"Evaluations found for genome g{best_gen:03d}/i{best_idx:02d}: {len(same_genome)}")
    if replicate_scores:
        lines.append(
            "Scores → max={mx}, min={mn}, mean={mean}, std≈{std}".format(
                mx=_format_float(max(replicate_scores), 3),
                mn=_format_float(min(replicate_scores), 3),
                mean=_format_float(sum(replicate_scores) / len(replicate_scores), 3),
                std=_format_float(np.std(replicate_scores), 3),
            )
        )
    if replicate_pnls:
        lines.append(
            "PnL → max={mx}, min={mn}, mean={mean}, std≈{std}".format(
                mx=_format_float(max(replicate_pnls), 1),
                mn=_format_float(min(replicate_pnls), 1),
                mean=_format_float(sum(replicate_pnls) / len(replicate_pnls), 1),
                std=_format_float(np.std(replicate_pnls), 1),
            )
        )
    split_repl = {}
    for entry in same_genome:
        split = entry.get("split", "train")
        bucket = split_repl.setdefault(split, {"scores": [], "pnls": []})
        if entry.get("score") is not None:
            bucket["scores"].append(float(entry["score"]))
        if entry.get("pnl") is not None:
            bucket["pnls"].append(float(entry["pnl"]))
    for split, data in split_repl.items():
        s_stats = _summary_stats(data.get("scores", []))
        p_stats = _summary_stats(data.get("pnls", []))
        lines.append(
            f"- {split} split → score mean={_format_float(s_stats.get('mean') if s_stats else None, 3)} +/-{_format_float(s_stats.get('ci95') if s_stats else None, 3)}, "
            f"pnl mean={_format_float(p_stats.get('mean') if p_stats else None, 1)} +/-{_format_float(p_stats.get('ci95') if p_stats else None, 1)}"
        )
    if replicate_dataset_counter:
        lines.append("Dataset coverage (replicates):")
        for dataset, count in replicate_dataset_counter.most_common():
            lines.append(f"- {dataset}: {count}")

    lines.append("")
    lines.append("### Dataset Emphasis Among Outliers")
    if dataset_top_counter:
        lines.append("Top-k focus:")
        for dataset, count in dataset_top_counter.most_common():
            lines.append(f"- {dataset}: {count} / {len(top)}")
    if dataset_all_counter:
        lines.append("Overall population:")
        for dataset, count in dataset_all_counter.most_common():
            lines.append(f"- {dataset}: {count}")

    lines.append("")
    lines.append("### Score Stability Checks")
    if train_mean is not None:
        lines.append(
            f"Train mean score = {_format_float(train_mean, 3)}; Train CVaR (worst {cvar_alpha:.2f}) = {_format_float(train_cvar, 3) if train_cvar is not None else 'NA'}"
        )
    if val_mean is not None:
        lines.append(f"Validation mean score = {_format_float(val_mean, 3)}")
    if train_scores_all:
        lines.append(
            f"Train worst five scores: {', '.join(_format_float(s, 3) for s in sorted(train_scores_all)[:5])}"
        )

    if split_metrics:
        lines.append("")
        lines.append("### Split-Level Performance Summary")
        lines.append("|split|n|score_mean+/-ci|score_q25–q75|pnl_mean+/-ci|inv_mean|")
        lines.append("|-----|--|---------------|------------|---------------|--------|")
        for split in sorted(split_metrics.keys()):
            metrics = split_metrics[split]
            score_stats = _summary_stats(metrics.get("scores", []))
            pnl_stats = _summary_stats(metrics.get("pnls", []))
            inv_stats = _summary_stats(metrics.get("inventory", []))
            lines.append(
                "|{split}|{n}|{score_ci}|{score_iqr}|{pnl_ci}|{inv_mean}|".format(
                    split=split,
                    n=score_stats.get("count") if score_stats else 0,
                    score_ci="{}+/-{}".format(
                        _format_float(score_stats.get("mean") if score_stats else None, 3),
                        _format_float(score_stats.get("ci95") if score_stats else None, 3),
                    ) if score_stats else "NA",
                    score_iqr="{}–{}".format(
                        _format_float(score_stats.get("q25") if score_stats else None, 3),
                        _format_float(score_stats.get("q75") if score_stats else None, 3),
                    ) if score_stats else "NA",
                    pnl_ci="{}+/-{}".format(
                        _format_float(pnl_stats.get("mean") if pnl_stats else None, 1),
                        _format_float(pnl_stats.get("ci95") if pnl_stats else None, 1),
                    ) if pnl_stats else "NA",
                    inv_mean=_format_float(inv_stats.get("mean") if inv_stats else None, 0),
                )
            )

    best_history_path = RUNS / "best_history.json"
    if best_history_path.exists():
        try:
            best_history = json.loads(best_history_path.read_text(encoding="utf-8"))
        except Exception as exc:
            best_history = []
            print(f"[warn] failed to parse {best_history_path}: {exc}", file=sys.stderr)
        if best_history:
            lines.append("")
            lines.append("### CMA-ES Trajectory Snapshot")
            last_entry = best_history[-1]
            lines.append(
                "Last recorded generation {gen}: best_score={best}, mean_score={mean}".format(
                    gen=last_entry.get("generation"),
                    best=_format_float(last_entry.get("best_score"), 3),
                    mean=_format_float(last_entry.get("mean_score"), 3),
                )
            )
            lines.append("Recorded best genome vector:")
            lines.append("```json")
            lines.append(json.dumps(last_entry.get("genome", []), indent=2))
            lines.append("```")

    output_path = Path(getattr(args, "analysis_output", RUNS / "outlier_analysis.md"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not quiet:
        print(f"[analysis] Wrote outlier report to {output_path}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small popsize/gens and 1×day/seed")
    ap.add_argument("--smoke", action="store_true", help="ultra-fast single generation for debugging")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="per-eval wall-clock timeout (s)")
    ap.add_argument("--seconds", type=int, default=DEFAULT_SECONDS, help="simulation seconds (<=0 falls back to minutes)")
    ap.add_argument("--max-evals", type=int, default=None, help="Total evaluation budget (stops early when reached)")
    ap.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP, help="Stop after this many stagnant generations (0 disables)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel worker processes for evaluations")
    ap.add_argument("--cma-seed", type=int, default=42)
    ap.add_argument("--score-mode", choices=["normalized", "raw"], default="normalized", help="Fitness function: normalized (PnL per second w/ normalized inventory penalty) or raw (legacy PnL penalty).")
    ap.add_argument("--loss-penalty", type=float, default=cfg.DEFAULT_LOSS_PENALTY, help="Extra penalty multiplier applied to large losses (as a fraction of start cash).")
    ap.add_argument("--validation-weight", type=float, default=cfg.DEFAULT_VALIDATION_WEIGHT, help="Weight applied to validation mean score when combining with train CVaR (0 disables).")
    ap.add_argument("--lob-file", default=str(DEFAULT_LOB_FILE), help="Path to a Nordic LOB dataset (txt). Leave empty to keep synthetic oracle.")
    ap.add_argument("--lob-dir", default=str(DEFAULT_LOB_DIR), help="Directory containing additional Nordic LOB files for rotation.")
    ap.add_argument("--lob-random", action="store_true", help="Randomize the dataset file per evaluation when --lob-dir has multiple files.")
    ap.add_argument("--robust-quantile", type=float, default=DEFAULT_ROBUST_QUANTILE, help="Fraction of worst training evaluations to average when scoring a genome (0 uses the mean).")
    ap.add_argument("--analyze", action="store_true", help="Analyze previous evaluations and exit without launching CMA-ES.")
    ap.add_argument("--analysis-top-k", type=int, default=10, help="Highlight the top-K evaluations in the analysis report.")
    ap.add_argument("--analysis-output", default=str(RUNS / "outlier_analysis.md"), help="Where to store the analysis report (Markdown).")
    ap.add_argument("--validate-genome", default=None, help="Path to a genome vector (JSON/CSV) or comma-separated list for hold-out validation runs.")
    ap.add_argument("--validate-output", default=None, help="Output directory for hold-out validation artifacts.")
    ap.add_argument("--skip-holdout", action="store_true", help="Skip automatic hold-out sweeps after CMA-ES completes.")
    ap.add_argument("--no-post-analysis", action="store_true", help="Skip automatic statistical analysis after CMA-ES completes.")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors in console output")
    ap.add_argument("--no-progress", action="store_true", help="Disable live progress bars")
    ap.add_argument("--gui", action="store_true", help="Launch a DearPyGui dashboard during optimisation (forces workers=1)")
    args = ap.parse_args()

    if getattr(args, "analyze", False):
        run_analysis(args)
        return

    args.timeout = args.timeout if args.timeout and args.timeout > 0 else DEFAULT_TIMEOUT
    args.seconds = args.seconds if args.seconds is not None else DEFAULT_SECONDS
    args.minutes = DEFAULT_MINUTES
    args.popsize = DEFAULT_POPSIZE
    args.gens = DEFAULT_GENS
    args.max_evals = args.max_evals if args.max_evals else DEFAULT_MAX_EVALS
    args.early_stop_patience = args.early_stop_patience if args.early_stop_patience is not None else DEFAULT_EARLY_STOP

    args.train_days = list(DEFAULT_TRAIN_DAYS)
    args.val_days = list(DEFAULT_VAL_DAYS)
    args.test_days = list(DEFAULT_TEST_DAYS)
    args.train_seeds = list(DEFAULT_TRAIN_SEEDS)
    args.val_seeds = list(DEFAULT_VAL_SEEDS)
    args.test_seeds = list(DEFAULT_TEST_SEEDS)
    args.days = sorted(set(args.train_days + args.val_days))
    args.seeds = sorted(set(args.train_seeds + args.val_seeds))

    args.robust_quantile = float(args.robust_quantile if args.robust_quantile is not None else DEFAULT_ROBUST_QUANTILE)
    args.robust_quantile = min(max(args.robust_quantile, 0.0), 1.0)
    args.loss_penalty = float(getattr(args, "loss_penalty", cfg.DEFAULT_LOSS_PENALTY))
    args.validation_weight = float(getattr(args, "validation_weight", cfg.DEFAULT_VALIDATION_WEIGHT))

    lob_candidates = _resolve_lob_files(args.lob_file, args.lob_dir)
    if not lob_candidates:
        lob_candidates = [Path(DEFAULT_LOB_FILE).resolve()]
    args.lob_files = lob_candidates

    train_ds, val_ds, test_ds = auto_dataset_split(args.lob_files)
    args.train_dataset_paths = train_ds or args.lob_files
    args.val_dataset_paths = val_ds or []
    args.test_dataset_paths = test_ds or []

    if not args.test_dataset_paths:
        args.test_days = []
        args.test_seeds = []

    half = max(1, len(args.train_dataset_paths) // 2)
    args.warmup_train_dataset_paths = args.train_dataset_paths[:half]
    if args.val_dataset_paths:
        args.warmup_val_dataset_paths = args.val_dataset_paths[:max(1, len(args.val_dataset_paths) // 2)] or args.val_dataset_paths
    else:
        args.warmup_val_dataset_paths = []

    combined_lob = list(dict.fromkeys(args.train_dataset_paths + args.val_dataset_paths + args.test_dataset_paths))
    if combined_lob:
        args.lob_files = combined_lob

    args.dataset_agg = FULL_DATASET_AGG
    args.dataset_agg_post = FULL_DATASET_AGG
    args.dataset_agg_switch_gen = WARMUP_GENERATIONS
    args.warmup_generations = WARMUP_GENERATIONS
    args.warmup_dataset_agg = WARMUP_DATASET_AGG

    if args.validate_genome:
        genome = _load_genome_spec(args.validate_genome)
        if genome is None:
            print(f"[validation] Unable to parse genome specification: {args.validate_genome}", file=sys.stderr)
        else:
            output_dir = args.validate_output or (RUNS / "holdout_validation_manual")
            run_holdout(
                args,
                genome,
                "validation",
                args.val_days,
                args.val_seeds,
                args.val_dataset_paths,
                output_dir=output_dir,
            )
        return

    args.thin_value = DEFAULT_THIN["ValueAgent"]
    args.thin_noise = DEFAULT_THIN["NoiseAgent"]
    args.thin_momentum = DEFAULT_THIN["MomentumAgent"]
    args.thin_execution = DEFAULT_THIN["ExecutionAgent"]
    args.inv_penalty = DEFAULT_INV_PENALTY
    args.drawdown_threshold = DEFAULT_DRAWDOWN_THRESHOLD
    args.drawdown_penalty = DEFAULT_DRAWDOWN_PENALTY
    args.drawdown_clip = DEFAULT_DRAWDOWN_CLIP
    args.score_trim = DEFAULT_SCORE_TRIM
    args.mm_start_cash = DEFAULT_START_CASH
    args.lob_freq = "100ms"
    args.lob_scale = 10000.0
    args.lob_offset = 2.0
    args.lob_no_monotonic = False
    args.lob_seed = None

    if not getattr(args, "lob_random", False):
        args.lob_random = True

    if getattr(args, "gui", False):
        print("[mm_cmaes] launching DearPyGui dashboard...", flush=True)
        from mmcore.gui import launch_gui  # local import to avoid dearpygui dependency when unused

        event_bus = EventBus()
        if getattr(args, "workers", 1) != 1:
            print("[mm_cmaes] GUI mode forces --workers=1 for compatibility with live updates.")
        args.workers = 1

        runner = threading.Thread(target=run_cmaes, args=(args, event_bus, None, None), daemon=True)
        runner.start()
        try:
            launch_gui(event_bus, runner)
        finally:
            runner.join()
        return

    from mmcore.tui import launch_tui

    ctx = multiprocessing.get_context("spawn")
    event_queue = ctx.Queue()
    workers = max(1, int(getattr(args, "workers", 1)))
    args.no_progress = True
    args.no_color = True

    runner = threading.Thread(target=run_cmaes, args=(args, None, event_queue, ctx), daemon=True)
    runner.start()
    try:
        launch_tui(event_queue, runner, workers, total_generations=args.gens)
    finally:
        if runner.is_alive():
            runner.join()
        event_queue.close()
        event_queue.join_thread()

    return


if __name__ == "__main__":
    main()
