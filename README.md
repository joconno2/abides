# Adaptive Market Maker CMA-ES Study

This repository combines the ABIDES discrete-event market simulator with a CMA-ES harness (`mmcore/optimizer.py`, surfaced via the thin CLI shim `mm_cmaes.py`) for tuning an adaptive market-making strategy. The project is organized as a teaching and research case study: we aim to surface robust, generalizable policies and document the full experimental protocol so that advanced undergraduates can reproduce, analyze, and stress-test the results without any hand-waving or AI hype.

## Repository Layout

- `abides_core/` — vanilla ABIDES snapshot (frozen at commit `c4bf157`), patched only to support dotted config imports and modern Pandas APIs.
- `mmcore/` — market-maker optimisation logic (CMA-ES driver, dataset helpers, shared utilities).
- `mm_configs/` — adaptive RMSC03 config wrapper layered on top of the stock ABIDES config.
- `mm_utils/` — bespoke helpers (e.g., Nordic LOB oracle) that extend ABIDES without modifying the vendored core.
- `historic_runs/` — timestamped experiment outputs (`mm_<timestamp>/…`). New runs append here automatically; previous `runs_mvp` artifacts were archived under the same hierarchy.
- `tools/` — local analysis scripts (e.g., `plot_mm_perf.py`).

## Experimental Outline

- **Environment.** ABIDES simulates a continuous double auction populated by value, noise, momentum, and execution agents. Our candidate market maker is an adaptive, parameter-rich agent exposed through a JSON configuration interface.
- **Data.** We draw limit-order-book streams from the Nordic dataset (`BenchmarkDatasets/…/Train_Dst_NoAuction_ZScore_CF_*.txt`). Each evaluation samples one file (optionally at random) and feeds it to ABIDES with 100 ms clocking and standard scaling.
- **Genome.** CMA-ES evolves an 8-dimensional vector that maps to participation rate, quote sizing, skew, ladder spacing, spread smoothing, cancel delay, inventory aversion, and max inventory multiplier. Bounds are documented inline in `mmcore/optimizer.py`.
- **Training/validation/test splits.** A deterministic shuffle (seed 1729) breaks the available Nordic feeds into train/validation/test groups (60/20/20). Training currently uses seeds {1–6} on day `20200603`, validation uses seeds {7, 8}, and the hold-out suite uses seeds {9, 10} on `20200604`. Every genome is still evaluated across **all** configured scenarios each generation; the best genome is replayed on validation and test after optimisation to keep those sets pristine.
- **Fitness.** Each evaluation collects PnL, inventory usage, and score components. Training fitness is the Conditional Value-at-Risk (CVaR) of the training scores (worst `robust_quantile`, trimmed by `score_trim`), so a genome must survive adverse samples to rank well. A validation mean term (default weight 0.25) is blended in, and large realised losses incur an additional penalty scaled by `--loss-penalty`.
- **Reporting.** Every evaluation writes a `summary.json` (parameters, metrics, split label). Once CMA-ES terminates we automatically synthesize `historic_runs/mm_*/outlier_analysis.md`, a Markdown report containing:
  - top-scoring evaluations (with split, dataset, and parameter context),
  - train/validation descriptive statistics with 95 % confidence intervals,
  - replication summaries for the best genome, and
  - CMA-ES convergence snapshots.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The simulator depends on a POSIX environment, Python ≥3.9, and the `cma` package.
The optional GUI (`--gui`) relies on DearPyGui, which ships with `requirements.txt`.

## Running the Study

### Quick smoke test

```bash
python3 mm_cmaes.py --quick --workers 2
```

Quick mode reduces the population, generations, and scenario grid to confirm the pipeline without long runtimes. Outputs appear under `historic_runs/mm_*/evals/` (a fresh timestamped directory per invocation) and the post-run analysis is refreshed automatically.
The terminal monitor opens automatically; press `q` after the run finishes to return to the shell.

### Live dashboard

```bash
python3 mm_cmaes.py --gui
```

`--gui` launches a DearPyGui dashboard that streams evaluation progress, per-scenario outcomes, and generation scores in real time. The GUI forces the optimiser to run with a single worker so events can be emitted synchronously.

### Terminal TUI

```bash
python3 mm_cmaes.py --workers 12
```

The terminal dashboard is now the default CLI output. It renders a generation × worker grid (with `▶` during execution, `✓` on success, `✗` on failure) alongside detailed per-worker metrics. Press `q` to exit once the optimisation completes.

### Full experiment (default settings)

```bash
python3 mm_cmaes.py --workers 12 --max-evals 144
```

Key options:

- `--workers`, `--max-evals`, `--seconds`: control runtime parallelism, budget, and per-sim duration.
- `--robust-quantile`: fraction of worst training outcomes per dataset averaged when scoring a genome (defaults to 0.25).
- `--loss-penalty`: additional weight applied to large negative PnL to discourage fragile genomes (defaults to 10.0).
- `--validation-weight`: multiplier for the validation mean score blended into the training CVaR objective (defaults to 0.25).
- `--lob-dir` / `--lob-file`: override the default Nordic dataset directory or seed file. `--lob-random` randomizes file selection when a split lacks dataset overrides.
- `--validate-genome`: run a saved genome across the validation lattice and emit a standalone report without rerunning CMA-ES.
- `--skip-holdout`: skip the automatic validation/test sweeps after CMA-ES completes.
- `--no-post-analysis`: skip the consolidated Markdown report if you prefer custom processing.

The harness automatically shuffles the available CF files (seed 1729) into 60 % training, 20 % validation, and 20 % test sets. Edit the constants near the top of `mmcore/optimizer.py` (`TRAIN_DATASET_COUNT`, `VAL_DATASET_COUNT`, etc.) if you need a different partition.

All other hyperparameters (population, generations, timeouts, penalties) remain configurable via command-line flags; see `python3 mm_cmaes.py --help` for the complete list.

## Output Anatomy

- `historic_runs/mm_*/evals/gXXX_iYY_split=…/summary.json` — per-scenario metrics including score components, raw PnL, inventory statistics, and the exact genome.
- `historic_runs/mm_*/best_history.json` — best genome tracked by CMA-ES after each generation.
- `historic_runs/mm_*/outlier_analysis.md` — consolidated statistical briefing, suitable for inclusion in lab reports.
- `historic_runs/mm_*/holdout_validation/` & `historic_runs/mm_*/holdout_test/` — automatic post-run sweeps of the best genome on the validation and test lattices (created unless `--skip-holdout` is supplied).
- `latest_output.ini` — rolling console stream from the latest optimization session.

The Markdown report is intentionally verbose: it discloses the distribution of outcomes across train/validation splits, highlights which datasets dominate the outlier tail, and tabulates the CMA-ES trajectory. Students can quote or extend these sections directly in their write-ups.

## Fitness Function Details

For a single ABIDES simulation with reward `pnl` and absolute inventory `inv_abs`:

1. Normalize profit: `pnl_normalized = pnl / start_cash` and `log_return = log1p(pnl_normalized)`.
2. Estimate per-second return and impose a drawdown penalty if the normalized rate falls below `-drawdown_threshold`.
3. Penalize inventory by `inv_penalty * (inv_abs / start_cash)`.
4. Combine as `score = log_return - inventory_penalty - drawdown_penalty`, clipping to `-drawdown_clip` on the downside.
5. Mark invalid runs (timeouts, missing PnL, configuration errors) with a sentinel score of `-1_000_000`.

Across the training split, scores are sorted ascending and the worst `robust_quantile` share is averaged. This approximate CVaR objective asks CMA-ES to produce strategies whose *worst* training scenarios are still acceptable.

## CMA-ES Recap

- CMA-ES maintains a multivariate normal search distribution over genomes.
- Each generation samples a population, evaluates fitness, ranks candidates, and updates the mean and covariance to favor better points.
- Step-size control adapts the overall search radius, while covariance adaptation captures interactions between genes (e.g., how participation rate and cancel delay co-vary in strong strategies).
- We seed CMA-ES at the mid-point of each gene’s admissible range and select a population of 12 by default, yielding 144 candidate evaluations over 12 generations unless early stopping triggers.
- Parameter bounds are regularised: POV proxies cap near 35 %, ladder spacing bottoms out at 6 ticks, and inventory multipliers at 3× to discourage runaway exposure before optimisation even begins.
- A short warm-up phase (controlled by constants near the top of `mm_cmaes.py`) runs on a pruned dataset list with a gentler aggregation before switching to the full worst-case objective.

Useful references: Hansen (2006) “The CMA Evolution Strategy” and Auger & Hansen (2011) “Theory of Evolution Strategies”.

## Example Result Snapshot

Extracted from `historic_runs/mm_*/outlier_analysis.md` (illustrative):

```json
Genome: [-33.30, 391.29, -129.16, 172.92, -59.03, 92.42, -21.61, 70.39]
Mapped parameters: {
  "pov": 0.015,
  "min_order_size": 391,
  "spread_alpha": 0.10,
  "cancel_limit_delay": 92,
  "inventory_risk_aversion": 0.10,
  "inventory_limit": 3128
}
```

The accompanying report quantifies how this genome behaves across training and validation seeds, including 95 % confidence intervals for returns and inventory usage.

## Warm-up & Hold-out Workflow

- **Warm-up generations** are built in: the first `WARMUP_GENERATIONS` (currently 2) run on a trimmed subset of the training/validation feeds using a mean aggregation. After that the optimiser switches to the full lattice with a worst-case (min) objective. Adjust the constants near the top of `mmcore/optimizer.py` if you need a different schedule.
- After optimisation, the harness automatically replays the best genome on both the validation and test lattices (unless `--skip-holdout` is passed), writing Markdown reports under `historic_runs/mm_*/holdout_validation/` and `historic_runs/mm_*/holdout_test/`. This ensures the final tables reference data that never influenced the optimisation process.
