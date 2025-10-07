# -*- coding: utf-8 -*-
r"""
mm_cmaes.py — CMA-ES harness for tuning a Market Maker in ABIDES.

What’s new in this build
------------------------
1) Broadcasts a wide set of AdaptiveMarketMaker knobs (pov, sizing, skew_beta,
   level spacing, spread alpha, cancel delay, wake frequency, backstop size)
   plus synonyms so downstream forks pick up changes reliably.
2) Prints the parsed PnL & |inventory| per eval so you immediately see variation.
3) Streams ABIDES output; captures partial output on timeout; writes tails.
4) Supports Nordic LOB dataset runs, including rotating/randomized files per eval
   for richer objectives.
5) Normalizes CMA-ES fitness (PnL per second with inventory penalty) by default,
   while still allowing the legacy raw score via --score-mode.
6) Parallelizes evaluations with --workers so you can spread the population across
   multiple processes (e.g., 12) during tuning.
7) Adds drawdown-aware penalties and tighter parameter bounds to avoid catastrophic
   strategies during optimization.

Quick smoke runs
----------------
$env:PYTHONPATH = "$PWD"
python .\mm_cmaes.py --quick

If scores/PnL don’t budge across evals, your MM isn’t consuming our knobs in
this fork. In that case we’ll switch the config to a simpler, fully tunable MM.
"""

import argparse
import math
import os
import sys
import json
import time
import subprocess
import threading
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

try:
    import cma
except Exception:
    print("Please install cma: pip install cma", file=sys.stderr)
    raise

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs_mvp"
EVALS = RUNS / "evals"
EVALS.mkdir(parents=True, exist_ok=True)
ABIDES_PY = str((ROOT / "abides.py").resolve())
DEFAULT_LOB_FILE = ROOT / "BenchmarkDatasets/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_1.txt"
DEFAULT_LOB_DIR = DEFAULT_LOB_FILE.parent

DEFAULT_SECONDS = 40
DEFAULT_TIMEOUT = 600
DEFAULT_MINUTES = 1
DEFAULT_POPSIZE = 12
DEFAULT_GENS = 6
DEFAULT_MAX_EVALS = DEFAULT_POPSIZE * DEFAULT_GENS
DEFAULT_EARLY_STOP = 3
DEFAULT_SEEDS = [1, 2, 3, 4]
DEFAULT_DAYS = ["20200603"]
DEFAULT_INV_PENALTY = 5.0
DEFAULT_DRAWDOWN_THRESHOLD = 0.05
DEFAULT_DRAWDOWN_PENALTY = 50.0
DEFAULT_DRAWDOWN_CLIP = 1.5
DEFAULT_START_CASH = 10_000_000.0
DEFAULT_THIN = {
    "ValueAgent": 1,
    "NoiseAgent": 12,
    "MomentumAgent": 2,
    "ExecutionAgent": 1,
}

RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"

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
    [  1,   40],   # 0 → pov proxy (0.01–0.45)
    [ 50,  800],   # 1 → size (tighter to avoid runaway inventory)
    [  0,    6],   # 2 → skew beta (limit extreme skew)
    [  2,   15],   # 3 → level spacing
    [0.10, 0.85],  # 4 → spread alpha (avoid ultraslow/ultrafast updates)
    [ 20,  250],   # 5 → cancel delay (ns)
    [0.10,  2.0],  # 6 → inventory risk aversion
    [1.00, 4.0],   # 7 → max inventory multiplier
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
    color = GREEN if score > 0 else RED if score < 0 else YELLOW
    return f"{color}{score:.6f}{RESET}"


def _progress_bar(current, total, width=24):
    if total <= 0:
        return "[?]"
    pct = min(1.0, max(0.0, current / total))
    filled = int(round(pct * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({pct * 100:5.1f}%)"

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


def _select_lob_path(args, eval_uid, combo_idx):
    files = getattr(args, "lob_files", None)
    if not files:
        return None, "synthetic"
    if len(files) == 1:
        path = files[0]
    else:
        combos = max(1, len(getattr(args, "days", [])) * len(getattr(args, "seeds", [])))
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
        "-c", "rmsc03",  # wrapper shim is expected in config/rmsc03.py
        "-t", "ABM",
        "-d", day,
        "-s", str(seed),
        "-l", label,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
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

def _evaluate_once(genome, gen, idx, day, seed, args, combo_idx):
    label = f"mm_mvp_mm_g{gen:03d}_i{idx:02d}_d{day[-1]}_s{seed}"
    out_dir = EVALS / f"g{gen:03d}_i{idx:02d}_d{day[-1]}_s{seed}"
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

    dataset_path, dataset_tag = _select_lob_path(args, eval_uid, combo_idx)

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

    with open(out_dir / "cmdline.txt", "w", encoding="utf-8") as f:
        f.write(f"CMD: python -u abides.py -c rmsc03 -t ABM -d {day} -s {seed} -l {label}\n")
        f.write(f"PYTHONPATH={ROOT}\nMM_MVP_CFG={cfg_path}\n")

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
            base = pnl_per_second if pnl_per_second is not None else float(pnl)
            penalty += drawdown_penalty
        else:
            penalty = args.inv_penalty * float(inv_abs)
            base = float(pnl)
            penalty += drawdown_penalty
        score = float(base) - float(penalty)
        clip_limit = getattr(args, "drawdown_clip", None)
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
            "drawdown_penalty": drawdown_penalty,
            "clip_limit": clip_limit,
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

    print(
        f"[eval g{gen:03d} i{idx:02d} d{day[-1]} s{seed}] "
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
        "mm_params": mm_info,
        "mm_summary": mm_summary,
        "mm_applied": [(str(a), float(v) if isinstance(v, (int, float)) else v) for a, v in mm_attrs_applied] if mm_attrs_applied else [],
        "lob_dataset": lob_cfg,
    }
    try:
        (out_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[warn] failed to write summary.json: {exc}", file=sys.stderr)

    return score

def _evaluate_mean(genome, gen, idx, args):
    scores = []
    combo_idx = 0
    for day in args.days:
        for seed in args.seeds:
            scores.append(_evaluate_once(genome, gen, idx, day, seed, args, combo_idx))
            combo_idx += 1
    return float(np.mean(scores))

# ---------- CMA-ES loop ----------
def run_cmaes(args):
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
    print("[mm_cmaes] Cleared previous evaluation outputs.")

    args.lob_files = _resolve_lob_files(args.lob_file, args.lob_dir)

    if getattr(args, "lob_random", False) and not args.lob_files:
        print("[warn] --lob-random requested but no dataset files were discovered.", file=sys.stderr)

    smoke = getattr(args, "smoke", False)
    if smoke:
        args.quick = True

    if args.quick:
        popsize = 4
        gens = 2
        args.days = [args.days[0]]
        args.seeds = [args.seeds[0]]
        if args.seconds <= 0:
            args.seconds = DEFAULT_SECONDS
        args.max_evals = min(args.max_evals, popsize * gens * len(args.days) * len(args.seeds))
    else:
        popsize = args.popsize
        gens = args.gens

    if smoke:
        popsize = min(popsize, 2)
        gens = min(gens, 1)
        args.days = [args.days[0]]
        args.seeds = [args.seeds[0]]
        if args.seconds <= 0:
            args.seconds = 5
        else:
            args.seconds = min(args.seconds, 5)
        args.minutes = min(args.minutes, 1)
        args.max_evals = min(args.max_evals, popsize * gens * len(args.days) * len(args.seeds))

    eval_budget = args.max_evals if getattr(args, "max_evals", None) else None
    if eval_budget and eval_budget > 0:
        max_gens = max(1, math.ceil(eval_budget / popsize))
        gens = min(gens, max_gens)

    args.popsize_effective = int(popsize)

    es = cma.CMAEvolutionStrategy(x0, sigma0, {"popsize": int(popsize), "seed": args.cma_seed})

    evals_done = 0
    best_score_so_far = float("-inf")
    stagnant_gens = 0
    patience = max(0, getattr(args, "early_stop_patience", 0))
    workers = max(1, min(int(getattr(args, "workers", 1)), popsize))
    best_history = []

    executor = None
    if workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers)

    try:
        for gen in range(1, gens + 1):
            X = es.ask()
            population = list(enumerate(X))
            if eval_budget:
                remaining = max(0, eval_budget - evals_done)
                if remaining <= 0:
                    print("Evaluation budget exhausted before generation start.")
                    break
                population = population[:remaining]

            if not population:
                break

            F_map = {}
            total = len(population)
            show_progress = (not getattr(args, "no_progress", False)) and sys.stdout.isatty()
            completed = 0
            color_prefix = BLUE if _use_color(args) else ""
            color_reset = RESET if _use_color(args) else ""

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
                    score = _evaluate_mean(x, gen, i, args)
                    F_map[i] = -score
                    evals_done += 1
                    completed += 1
                    _print_progress()
            else:
                futures = {executor.submit(_evaluate_mean, X[i], gen, i, args): i for i, _ in population}
                for future in as_completed(futures):
                    i = futures[future]
                    score = future.result()
                    F_map[i] = -score
                    evals_done += 1
                    completed += 1
                    _print_progress()

            evaluated_indices = [i for i, _ in population if i in F_map]
            if not evaluated_indices:
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

            if gen_best_score > best_score_so_far + 1e-9:
                best_score_so_far = gen_best_score
                stagnant_gens = 0
            else:
                stagnant_gens += 1
                if patience and stagnant_gens >= patience:
                    print(f"Early stopping after {gen} generations (patience={patience}).")
                    break

            if eval_budget and evals_done >= eval_budget:
                print("Evaluation budget exhausted; stopping CMA-ES loop.")
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    try:
        (RUNS / "best_history.json").write_text(json.dumps(best_history, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[warn] failed to write best_history.json: {exc}", file=sys.stderr)

    print(f"Done. Evaluations completed: {evals_done}. Per-eval files in runs_mvp/evals/.")

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
    ap.add_argument("--lob-file", default=str(DEFAULT_LOB_FILE), help="Path to a Nordic LOB dataset (txt). Leave empty to keep synthetic oracle.")
    ap.add_argument("--lob-dir", default=str(DEFAULT_LOB_DIR), help="Directory containing additional Nordic LOB files for rotation.")
    ap.add_argument("--lob-random", action="store_true", help="Randomize the dataset file per evaluation when --lob-dir has multiple files.")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors in console output")
    ap.add_argument("--no-progress", action="store_true", help="Disable live progress bars")
    args = ap.parse_args()

    args.timeout = args.timeout if args.timeout and args.timeout > 0 else DEFAULT_TIMEOUT
    args.seconds = args.seconds if args.seconds is not None else DEFAULT_SECONDS
    args.minutes = DEFAULT_MINUTES
    args.popsize = DEFAULT_POPSIZE
    args.gens = DEFAULT_GENS
    args.max_evals = args.max_evals if args.max_evals else DEFAULT_MAX_EVALS
    args.early_stop_patience = args.early_stop_patience if args.early_stop_patience is not None else DEFAULT_EARLY_STOP
    args.days = list(DEFAULT_DAYS)
    args.seeds = list(DEFAULT_SEEDS)
    args.thin_value = DEFAULT_THIN["ValueAgent"]
    args.thin_noise = DEFAULT_THIN["NoiseAgent"]
    args.thin_momentum = DEFAULT_THIN["MomentumAgent"]
    args.thin_execution = DEFAULT_THIN["ExecutionAgent"]
    args.inv_penalty = DEFAULT_INV_PENALTY
    args.drawdown_threshold = DEFAULT_DRAWDOWN_THRESHOLD
    args.drawdown_penalty = DEFAULT_DRAWDOWN_PENALTY
    args.drawdown_clip = DEFAULT_DRAWDOWN_CLIP
    args.mm_start_cash = DEFAULT_START_CASH
    args.lob_freq = "100ms"
    args.lob_scale = 10000.0
    args.lob_offset = 2.0
    args.lob_no_monotonic = False
    args.lob_seed = None

    run_cmaes(args)

if __name__ == "__main__":
    main()
