"""Shared event definitions and dispatch utilities for the CMA-ES GUI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Any, List, Optional


@dataclass
class EvaluationResult:
    """Outcome of a single ABIDES evaluation."""

    label: str
    generation: int
    individual_index: int
    split: str
    day: str
    seed: int
    dataset: Optional[str]
    score: float
    pnl: Optional[float]
    inventory_abs: float
    reason: str
    duration: float
    start_time: float
    end_time: float
    mm_summary: str
    genome_vector: Optional[List[float]] = None
    score_components: Optional[dict] = None


@dataclass
class EvaluationStartEvent:
    label: str
    generation: int
    individual_index: int
    split: str
    day: str
    seed: int
    dataset_hint: Optional[str]
    timestamp: float


@dataclass
class EvaluationCompleteEvent:
    result: EvaluationResult


@dataclass
class GenerationSummaryEvent:
    generation: int
    best_score: float
    mean_score: float
    evaluations_completed: int
    timestamp: float


@dataclass
class RunCompleteEvent:
    evaluations_completed: int
    duration: float
    timestamp: float


class EventBus:
    """Thread-safe publish/subscribe queue used for GUI updates."""

    def __init__(self) -> None:
        self._queue: Queue = Queue()
        self._start_time = time.time()

    def publish(self, event: Any) -> None:
        if event is None:
            return
        self._queue.put(event)

    def subscribe(self) -> Queue:
        return self._queue

    def is_idle(self) -> bool:
        return self._queue.empty()

    @property
    def start_time(self) -> float:
        return self._start_time


def drain_queue(queue: Queue) -> List[Any]:
    """Return all currently queued events without blocking."""

    items: List[Any] = []
    while True:
        try:
            items.append(queue.get_nowait())
        except Empty:
            break
    return items
