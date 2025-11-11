"""Plot CMA-ES evaluation performance (PnL & Sharpe) across generations.

Usage:
    python tools/plot_mm_perf.py [--eval-root <run_dir>/evals] [--out mm_perf.png]

The script expects each evaluation directory under ``--eval-root`` to contain a
``summary.json`` file (written by ``mmcore.optimizer``).  It aggregates
per-generation statistics and produces a Matplotlib figure with average vs.
best PnL and the corresponding Sharpe ratio.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import mm_config as cfg


def _default_eval_root() -> Path:
    runs_root = Path(cfg.RUNS_ROOT)
    if not runs_root.exists():
        return runs_root / "evals"
    candidates = sorted((p for p in runs_root.iterdir() if p.is_dir()), reverse=True)
    for candidate in candidates:
        eval_dir = candidate / "evals"
        if eval_dir.is_dir():
            return eval_dir
    return runs_root / "evals"


def load_summaries(eval_root: Path) -> pd.DataFrame:
    records = []
    for summary_path in eval_root.rglob("summary.json"):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            records.append(data)
        except Exception:
            continue
    if not records:
        raise FileNotFoundError(
            f"No summary.json files found under {eval_root}. Did you re-run mm_cmaes.py after the latest changes?"
        )
    df = pd.DataFrame(records)
    df.sort_values(["generation", "pop_index"], inplace=True)
    return df


def compute_generation_stats(df: pd.DataFrame):
    metric = "pnl_per_second" if "pnl_per_second" in df.columns and df["pnl_per_second"].notna().any() else "pnl"
    grouped = df.groupby("generation", group_keys=False)

    stats = grouped.agg(
        mean_metric=(metric, "mean"),
        best_metric=(metric, "max"),
        mean_score=("score", "mean"),
        best_score=("score", "max"),
    ).reset_index()

    sharpe_series = grouped[metric].apply(
        lambda x: float("nan") if x.empty else (x.mean() / x.std(ddof=0) if x.std(ddof=0) else float("nan"))
    )
    stats = stats.merge(sharpe_series.rename("sharpe"), left_on="generation", right_index=True, how="left")
    return stats, metric


def plot_stats(stats: pd.DataFrame, metric: str, out_path: Path):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    metric_label = "PnL / sec" if metric == "pnl_per_second" else "PnL"

    axes[0].plot(stats["generation"], stats["mean_metric"], label=f"Mean {metric_label}", linewidth=2)
    axes[0].plot(
        stats["generation"],
        stats["best_metric"],
        label=f"Best {metric_label}",
        linewidth=2,
        marker="o",
        markersize=6,
    )
    axes[0].set_ylabel(metric_label)
    axes[0].set_title("Market Maker Performance by Generation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(stats["generation"], stats["sharpe"], color="C2", linewidth=2)
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Sharpe (PnL)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_path.resolve()
    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot MM CMA-ES performance statistics")
    ap.add_argument(
        "--eval-root",
        default=str(_default_eval_root()),
        help="Root directory containing evaluation outputs (defaults to the latest run)",
    )
    ap.add_argument("--out", default="mm_performance.png", help="Output PNG path")
    args = ap.parse_args()

    eval_root = Path(args.eval_root).expanduser().resolve()
    df = load_summaries(eval_root)
    stats, metric = compute_generation_stats(df)

    pretty = stats.rename(columns={"mean_metric": f"mean_{metric}", "best_metric": f"best_{metric}"})
    print("Generation statistics:\n", pretty.to_string(index=False))
    out_path = Path(args.out)
    plot_stats(stats, metric, out_path)
    csv_path = out_path.with_suffix(".csv")
    pretty.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
