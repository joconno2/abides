"""Dataset utilities for the market-maker CMA workflow."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mm_config as cfg


def resolve_dataset_paths(names: Iterable[str], base_dir: Path | str) -> List[Path]:
    """Return a list of validated dataset paths.

    Resolution falls back to `base_dir / name` when `name` is relative. Missing
    files are skipped silently.
    """
    paths: List[Path] = []
    base = Path(base_dir)
    for name in names or []:
        if not name:
            continue
        candidate = Path(name)
        if not candidate.is_file():
            candidate = base / name
        if candidate.is_file():
            try:
                paths.append(candidate.resolve())
            except FileNotFoundError:
                continue
    return paths


def auto_dataset_split(files: Sequence[Path | str]) -> Tuple[List[Path], List[Path], List[Path]]:
    """Shuffle assets into train/validation/test partitions."""
    file_paths = [Path(f).resolve() for f in files]
    file_paths = sorted(dict.fromkeys(file_paths))
    if not file_paths:
        return [], [], []

    rng = random.Random(cfg.DATASET_SPLIT_SEED)
    shuffled = file_paths[:]
    rng.shuffle(shuffled)

    train = shuffled[: cfg.TRAIN_DATASET_COUNT] or shuffled[:]
    remainder = shuffled[len(train):]
    val = remainder[: cfg.VAL_DATASET_COUNT]
    remainder = remainder[len(val):]
    test = remainder[: cfg.TEST_DATASET_COUNT]

    if not val and len(train) > 1:
        val = [train.pop()]
    if not test and remainder:
        test = remainder[:]
    if not test and len(train) > len(val):
        test = [train.pop()]

    return train, val, test


def make_split_combos(
    split: str,
    days: Sequence[str],
    seeds: Sequence[int],
    datasets: Sequence[Path | str],
) -> List[Dict[str, Optional[Path]]]:
    """Build evaluation tuples for the requested split."""
    combos: List[Dict[str, Optional[Path]]] = []
    day_list = list(days or [])
    seed_list = list(seeds or [])
    dataset_list = list(datasets or [])
    if not day_list or not seed_list:
        return combos
    resolved_datasets = dataset_list if dataset_list else [None]
    for day in day_list:
        for seed in seed_list:
            for dataset in resolved_datasets:
                dataset_obj = Path(dataset) if dataset else None
                combos.append(
                    {
                        "split": split,
                        "day": day,
                        "seed": seed,
                        "dataset": dataset_obj,
                    }
                )
    return combos


def prepare_combo_plan(
    train_days: Sequence[str],
    train_seeds: Sequence[int],
    train_datasets: Sequence[Path | str],
    val_days: Sequence[str],
    val_seeds: Sequence[int],
    val_datasets: Sequence[Path | str],
) -> List[Dict[str, Optional[Path]]]:
    """Combine training and validation schedules for one optimisation round."""
    combos: List[Dict[str, Optional[Path]]] = []
    combos.extend(make_split_combos("train", train_days, train_seeds, train_datasets))
    combos.extend(make_split_combos("validation", val_days, val_seeds, val_datasets))
    if not combos:
        combos.append(
            {
                "split": "train",
                "day": cfg.DEFAULT_DAYS[0],
                "seed": cfg.DEFAULT_TRAIN_SEEDS[0],
                "dataset": None,
            }
        )
    return combos


__all__ = [
    "resolve_dataset_paths",
    "auto_dataset_split",
    "make_split_combos",
    "prepare_combo_plan",
]
