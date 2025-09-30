"""
mm_cmaes.py — CMA-ES for evolving a Market Maker in ABIDES.

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
except Exception as e:
    print("Please install cma: pip install cma", file=sys.stderr)
    raise

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs_mvp"
EVALS = RUNS / "evals"
EVALS.mkdir(parents=True, exist_ok=True)

ABIDES_PY = str((ROOT / "abides.py").resolve())  # ABIDES entry point

# ---------- Params ----------
# [spread_bps, quote_size, inv_aversion, skew_gain, cancel_thresh_bps, max_inventory]
BOUNDS = np.array([
    [  1,  50],    # spread_bps
    [ 50, 1000],   # quote_size
    [  0,  10],    # inv_aversion
    [  0,  10],    # skew_gain
    [  0,  50],    # cancel_thresh_bps
    [100, 5000],   # max_inventory
], dtype=float)

MID = BOUNDS.mean(axis=1)
SIGMA0 = float((BOUNDS[:,1] - BOUNDS[:,0]).mean() / 3.0)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def genome_to_mm_params(x):
    x = np.asarray(x, dtype=float)
    x = np.array([clamp(v, lo, hi) for v,(lo,hi) in zip(x, BOUNDS)])
    return {
        "spread_bps":        int(round(x[0])),
        "quote_size":        int(round(x[1])),
        "inv_aversion":      float(x[2]),
        "skew_gain":         float(x[3]),
        "cancel_thresh_bps": int(round(x[4])),
        "max_inventory":     int(round(x[5])),
    }

# ---------- stdout parsing ----------
# Number with optional commas/decimals:  -12,345.67  or  12345  etc.
NUM = r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?"

# 1) Direct summary block:
RE_HEADER = re.compile(r"^\s*Mean ending value by agent type\s*:\s*$", re.IGNORECASE)
RE_AGENT_LINE = re.compile(rf"^\s*(.+?)\s*:\s*({NUM})\s*$")

# 2) Specific single-line shortcut (covers many forks):
RE_PNL_INLINE = re.compile(
    rf"^\s*(?:Adaptive.*MarketMaker|ADAPTIVE[_ ]POV[_ ]MARKET[_ ]MAKER(?:[_ ]AGENT)?)\s*:\s*({NUM})\s*$",
    re.IGNORECASE
)

# 3) Fallback: per-agent holdings with marked-to-market
RE_MM_HOLDINGS = re.compile(
    r"^Final holdings for .*MARKET.*MAKER.*\{\s*ABM:\s*(-?\d+)\b.*?Marked to market:\s*(" + NUM + r")\s*$",
    re.IGNORECASE
)

def _num(s):
    return float(str(s).replace(",", ""))

def parse_pnl_from_summary(so: str):
    """Look for the summary block and return dict {agent_type: pnl}."""
    lines = so.splitlines()
    pnl_dict = {}
    in_block = False
    for ln in lines:
        if not in_block:
            if RE_HEADER.search(ln):
                in_block = True
            continue
        # end of block if blank line or non "X: Y" line
        m = RE_AGENT_LINE.match(ln)
        if m:
            agent, val = m.group(1).strip(), _num(m.group(2))
            pnl_dict[agent] = val
        else:
            # likely end of block
            if pnl_dict:
                break
    return pnl_dict if pnl_dict else None

def parse_pnl(so: str, mm_start_cash: float):
    # Try the inline convenience first
    m = RE_PNL_INLINE.search(so)
    if m:
        return _num(m.group(1))

    # Try the full summary block
    pnl_block = parse_pnl_from_summary(so)
    if pnl_block:
        # find a key that looks like the MM
        # we allow many variants:
        for k, v in pnl_block.items():
            key_up = k.upper().replace(" ", "_")
            if "MARKET" in key_up and "MAKER" in key_up:
                return float(v)

    # Fallback: compute from holdings 'Marked to market'
    # Use the first MM line we find
    for ln in so.splitlines()[::-1]:  # scan from end (usually printed at end)
        m2 = RE_MM_HOLDINGS.search(ln)
        if m2:
            # inv = int(m2.group(1))  # not needed for PnL
            m2m = _num(m2.group(2))
            return float(m2m - mm_start_cash)

    return None

def parse_inventory_abs(so: str):
    # Just for penalty: look for the last MM holdings and read ABM qty
    for ln in so.splitlines()[::-1]:
        m = RE_MM_HOLDINGS.search(ln)
        if m:
            try:
                return abs(float(m.group(1)))
            except Exception:
                pass
    return None

# ---------- streaming runner ----------
def _reader_thread(pipe, acc):
    try:
        for line in iter(pipe.readline, ''):
            acc.append(line)
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass

def run_abides_streaming(label, seed, day, cfg_path: Path, timeout_s: int):
    cmd = [
        sys.executable, "-u", ABIDES_PY,
        "-c", "rmsc03",  # shimmed wrapper file (config/rmsc03.py)
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

    # Poll with timeout
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

    # Join readers
    t_so.join(timeout=1.0)
    t_se.join(timeout=1.0)

    dur = time.time() - t0
    rc = proc.returncode if proc.returncode is not None else 124
    tag = "timeout" if killed or rc == 124 else "ok"
    so = "".join(so_lines)
    se = "".join(se_lines)
    return rc, dur, so, se, tag

# ---------- per-eval ----------
def write_cfg(dir_path: Path, payload: dict):
    cfg_path = dir_path / "mvp_cfg.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return cfg_path

def _tail(text: str, n=40):
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if lines else ""

def evaluate_once(genome, gen, idx, day, seed, args):
    label = f"mm_mvp_mm_g{gen:03d}_i{idx:02d}_d{day[-1]}_s{seed}"
    out_dir = EVALS / f"g{gen:03d}_i{idx:02d}_d{day[-1]}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "mm_params": genome_to_mm_params(genome),
    }
    cfg_path = write_cfg(out_dir, payload)

    with open(out_dir / "cmdline.txt", "w", encoding="utf-8") as f:
        f.write(f"CMD: python -u abides.py -c rmsc03 -t ABM -d {day} -s {seed} -l {label}\n")
        f.write(f"PYTHONPATH={ROOT}\n")
        f.write(f"MM_MVP_CFG={cfg_path}\n")

    rc, dur, so, se, tag = run_abides_streaming(label, seed, day, cfg_path, timeout_s=args.timeout)

    # Save streams
    (out_dir / "cmd_stdout.txt").write_text(so or "", encoding="utf-8", errors="ignore")
    (out_dir / "cmd_stderr.txt").write_text(se or "", encoding="utf-8", errors="ignore")

    # Wrapper marker (should be next to cfg; the shim writes there)
    mm_marker = (out_dir / "wrapper_seen.txt").exists()

    # Parse fitness
    pnl = parse_pnl(so or "", args.mm_start_cash)
    inv_abs = parse_inventory_abs(so or "")
    inv_penalty = inv_abs if inv_abs is not None else 0.0

    # Determine reason and score
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
        score = float(pnl) - args.inv_penalty * float(inv_penalty)

    so_b = len(so.encode("utf-8")) if so else 0
    se_b = len(se.encode("utf-8")) if se else 0

    tail_note = ""
    if score <= -999999.9:
        tail = _tail(so or "", 40)
        (out_dir / "stdout_tail.txt").write_text(tail, encoding="utf-8", errors="ignore")
        tail_note = " tail=stdout_tail.txt"

    print(f"[eval g{gen:03d} i{idx:02d} d{day[-1]} s{seed}] rc={rc} dur={dur:.1f}s "
          f"score={score:.1f} mm_marker={int(mm_marker)} stdout={so_b}B stderr={se_b}B reason={reason}{tail_note}")

    return score

def evaluate_mean(genome, gen, idx, args):
    scores = []
    for day in args.days:
        for seed in args.seeds:
            s = evaluate_once(genome, gen, idx, day, seed, args)
            scores.append(s)
    return float(np.mean(scores))

# ---------- CMA-ES loop ----------
def run_cmaes(args):
    x0 = MID.copy()
    sigma0 = SIGMA0
    if args.quick:
        popsize = 4
        gens = 2
        args.days  = [args.days[0]]     # 1 day
        args.seeds = [args.seeds[0]]    # 1 seed
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
            f = -evaluate_mean(x, gen, i, args)  # CMA-ES minimizes
            F.append(f)
        es.tell(X, F)
        best_idx = int(np.argmin(F))
        best = X[best_idx]
        best_score = -F[best_idx]
        print(f"[gen {gen}/{gens}] best_score={best_score:.3f} genome={np.array2string(best, precision=6)}")

    print("Done. See per-eval folders under runs_mvp/evals/.")

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
    ap.add_argument("--mm-start-cash", type=float, default=10_000_000.0, help="used for fallback PnL from M2M")
    ap.add_argument("--days", nargs="+", default=["20200603"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--cma-seed", type=int, default=42)
    args = ap.parse_args()
    run_cmaes(args)

if __name__ == "__main__":
    main()
