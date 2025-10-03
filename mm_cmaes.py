# -*- coding: utf-8 -*-
r"""
mm_cmaes.py — CMA-ES harness for tuning a Market Maker in ABIDES.

What’s new in this build
------------------------
1) Sends a *redundant* set of MM params (synonyms) so AdaptivePOV & other MMs
   are more likely to accept them:
     - size: min_order_size, order_size, quote_size
     - risk: inventory_risk_aversion, inv_aversion
     - skew: skew_gain, skew
     - cancels: cancel_threshold_bps, cancel_thresh_bps
     - inventory cap: max_inventory, inventory_limit
     - participation: pov (derived from the first gene)
2) Prints the parsed PnL & |inventory| per eval so you immediately see variation.
3) Streams ABIDES output; captures partial output on timeout; writes tails.

Quick smoke runs
----------------
$env:PYTHONPATH = "$PWD"
python .\mm_cmaes.py --quick --timeout 180 --seconds 8

If scores/PnL don’t budge across evals, your MM isn’t consuming our knobs in
this fork. In that case we’ll switch the config to a simpler, fully tunable MM.
"""

import argparse
import os
import sys
import json
import time
import subprocess
import threading
import re
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

# ---------- genome & mapping ----------
# Genome dims (we keep 6 for continuity, but map to *many* agent attrs):
# 0: participation proxy (we map to 'pov' in [0.01, 0.50])
# 1: size proxy        (we map to min/order/quote size)
# 2: inv risk          (we map to inv risk aversion)
# 3: skew gain         (we map to skew/skew_gain)
# 4: cancel threshold  (bps)
# 5: max inventory     (units)
BOUNDS = np.array([
    [  1,   50],   # 0 → pov proxy
    [ 50, 1000],   # 1 → size
    [  0,   10],   # 2 → inv risk
    [  0,   10],   # 3 → skew
    [  0,   50],   # 4 → cancel bps
    [100, 5000],   # 5 → max inventory
], dtype=float)
MID = BOUNDS.mean(axis=1)
SIGMA0 = float((BOUNDS[:,1] - BOUNDS[:,0]).mean() / 3.0)

def _clamp(v, lo, hi):
    return float(max(lo, min(hi, v)))

def _map_genome_to_mm_params(x):
    x = np.asarray(x, dtype=float)
    x = np.array([_clamp(v, lo, hi) for v,(lo,hi) in zip(x, BOUNDS)])

    # derive a participation ratio from gene 0
    pov = _clamp(0.005 + x[0] / 100.0, 0.01, 0.50)  # 0.01–0.50

    size = int(round(x[1]))
    inv_risk = float(x[2])
    skew = float(x[3])
    cancel_bps = int(round(x[4]))
    max_inv = int(round(x[5]))

    # Send a broad set of synonyms so the agent accepts *something*.
    mm = {
        # participation
        "pov": pov,
        "participation_rate": pov,

        # size knobs (many forks choose one of these)
        "min_order_size": size,
        "order_size": size,
        "quote_size": size,

        # inventory aversion / risk
        "inventory_risk_aversion": inv_risk,
        "inv_aversion": inv_risk,

        # skew/intensity
        "skew_gain": skew,
        "skew": skew,

        # cancel thresholds (bps)
        "cancel_threshold_bps": cancel_bps,
        "cancel_thresh_bps": cancel_bps,

        # inventory caps
        "max_inventory": max_inv,
        "inventory_limit": max_inv,
    }
    return mm

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

def _evaluate_once(genome, gen, idx, day, seed, args):
    label = f"mm_mvp_mm_g{gen:03d}_i{idx:02d}_d{day[-1]}_s{seed}"
    out_dir = EVALS / f"g{gen:03d}_i{idx:02d}_d{day[-1]}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mm_params = _map_genome_to_mm_params(genome)

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
    cfg_path = _write_cfg(out_dir, payload)

    with open(out_dir / "cmdline.txt", "w", encoding="utf-8") as f:
        f.write(f"CMD: python -u abides.py -c rmsc03 -t ABM -d {day} -s {seed} -l {label}\n")
        f.write(f"PYTHONPATH={ROOT}\nMM_MVP_CFG={cfg_path}\n")

    rc, dur, so, se, tag = _run_abides(label, seed, day, cfg_path, timeout_s=args.timeout)

    (out_dir / "cmd_stdout.txt").write_text(so or "", encoding="utf-8", errors="ignore")
    (out_dir / "cmd_stderr.txt").write_text(se or "", encoding="utf-8", errors="ignore")

    mm_marker = (out_dir / "wrapper_seen.txt").exists() or (ROOT / "wrapper_seen.txt").exists()

    pnl, inv_abs = _parse_pnl_and_inv(so or "", args.mm_start_cash)
    inv_abs = inv_abs if inv_abs is not None else 0.0

    reason = tag
    if rc != 0 and tag != "timeout":
        reason = f"rc={rc}"
    if not mm_marker:
        reason = "wrapper_not_seen"
    if pnl is None and reason == "ok":
        reason = "pnl_not_found"

    if (rc != 0) or (not mm_marker) or (pnl is None):
        score = -1_000_000.0
    else:
        score = float(pnl) - args.inv_penalty * float(inv_abs)

    # Helpful debug: show the parsed numbers & a short tail when failing.
    if score <= -999999.9:
        (out_dir / "stdout_tail.txt").write_text(_tail(so, 40), encoding="utf-8", errors="ignore")
        tail_note = " tail=stdout_tail.txt"
    else:
        tail_note = ""

    so_b = len((so or "").encode("utf-8"))
    se_b = len((se or "").encode("utf-8"))
    print(
        f"[eval g{gen:03d} i{idx:02d} d{day[-1]} s{seed}] "
        f"rc={rc} dur={dur:.1f}s score={score:.1f} mm_marker={int(mm_marker)} "
        f"pnl={pnl if pnl is not None else 'NA'} inv_abs={inv_abs:.0f} "
        f"stdout={so_b}B stderr={se_b}B reason={reason}{tail_note}"
    )

    return score

def _evaluate_mean(genome, gen, idx, args):
    scores = []
    for day in args.days:
        for seed in args.seeds:
            scores.append(_evaluate_once(genome, gen, idx, day, seed, args))
    return float(np.mean(scores))

# ---------- CMA-ES loop ----------
def run_cmaes(args):
    x0 = MID.copy()
    sigma0 = SIGMA0

    if args.quick:
        popsize = 4
        gens = 2
        args.days = [args.days[0]]
        args.seeds = [args.seeds[0]]
        if args.seconds <= 0:
            args.seconds = 20
    else:
        popsize = args.popsize
        gens = args.gens

    es = cma.CMAEvolutionStrategy(x0, sigma0, {"popsize": int(popsize), "seed": args.cma_seed})

    for gen in range(1, gens + 1):
        X = es.ask()
        F = []
        for i, x in enumerate(X):
            f = -_evaluate_mean(x, gen, i, args)  # CMA-ES minimizes
            F.append(f)
        es.tell(X, F)
        best_idx = int(np.argmin(F))
        best = X[best_idx]
        best_score = -F[best_idx]
        print(f"[gen {gen}/{gens}] best_score={best_score:.3f} genome={np.array2string(best, precision=6)}")

    print("Done. Per-eval files in runs_mvp/evals/.")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small popsize/gens and 1×day/seed")
    ap.add_argument("--timeout", type=int, default=180, help="per-eval wall-clock timeout (s)")
    ap.add_argument("--seconds", type=int, default=20, help="session seconds (if >0 overrides minutes)")
    ap.add_argument("--minutes", type=int, default=1, help="session minutes (used if seconds<=0)")
    ap.add_argument("--thin-value", type=int, default=1)
    ap.add_argument("--thin-noise", type=int, default=1)
    ap.add_argument("--thin-momentum", type=int, default=0)
    ap.add_argument("--thin-execution", type=int, default=0)
    ap.add_argument("--inv-penalty", type=float, default=50.0)
    ap.add_argument("--mm-start-cash", type=float, default=10_000_000.0, help="fallback PnL uses this")
    ap.add_argument("--days", nargs="+", default=["20200603"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--cma-seed", type=int, default=42)
    args = ap.parse_args()
    run_cmaes(args)

if __name__ == "__main__":
    main()
