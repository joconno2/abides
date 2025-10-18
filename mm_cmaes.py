"""Backward-compatible entrypoint for the MM CMA-ES harness."""

from mmcore.optimizer import main, run_cmaes, run_holdout  # re-export public API

__all__ = ["main", "run_cmaes", "run_holdout"]


if __name__ == "__main__":
    main()
