"""Curses-based TUI monitor for the CMA-ES optimiser."""

from __future__ import annotations

import curses
import time
from dataclasses import dataclass
from queue import Empty
from typing import Dict, Optional


@dataclass
class WorkerSlot:
    label: Optional[str] = None
    status: str = "idle"
    dataset: str = ""
    split: str = ""
    day: str = ""
    seed: Optional[int] = None
    generation: Optional[int] = None
    individual_index: Optional[int] = None
    start_time: float = 0.0
    last_update: float = 0.0
    score: Optional[float] = None
    pnl: Optional[float] = None
    duration: float = 0.0
    reason: str = ""

    def reset(self, timestamp: float) -> None:
        self.label = None
        self.status = "idle"
        self.dataset = ""
        self.split = ""
        self.day = ""
        self.seed = None
        self.generation = None
        self.individual_index = None
        self.start_time = 0.0
        self.last_update = timestamp
        self.score = None
        self.pnl = None
        self.duration = 0.0
        self.reason = ""


def _safe_addstr(window, y: int, x: int, text: str, attr=0):
    max_y, max_x = window.getmaxyx()
    if 0 <= y < max_y:
        try:
            window.addstr(y, x, text[: max_x - x - 1], attr)
        except curses.error:
            pass


def _format_score(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"


def _format_pnl(value: Optional[float]) -> str:
    if value is None:
        return "--"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}k"
    return f"{value:.0f}"


def launch_tui(event_queue, run_thread, worker_count: int, total_generations: Optional[int] = None) -> None:
    def _wrapped(stdscr):
        _run_tui(stdscr, event_queue, run_thread, worker_count, total_generations)

    curses.wrapper(_wrapped)


def _assign_slot(workers, label_to_slot: Dict[str, int], label: str, now: float) -> int:
    idle = [i for i, slot in enumerate(workers) if slot.label is None]
    if idle:
        idx = idle[0]
    else:
        idx = min(range(len(workers)), key=lambda i: workers[i].last_update)
    label_to_slot[label] = idx
    workers[idx].reset(now)
    workers[idx].label = label
    return idx


def _run_tui(stdscr, event_queue, run_thread, worker_count: int, total_generations: Optional[int]) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    workers = [WorkerSlot(last_update=time.time()) for _ in range(worker_count)]
    label_to_slot: Dict[str, int] = {}

    generation_text = "Awaiting first generation..."
    run_summary = "Run in progress..."
    run_completed = False
    grid_status: Dict[tuple, str] = {}
    generation_order = []
    planned_generations = total_generations or 0

    while True:
        # Drain queue
        while True:
            try:
                event = event_queue.get_nowait()
            except Empty:
                break
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            now = time.time()
            if event_type == "meta":
                planned_generations = int(event.get("total_generations", planned_generations) or planned_generations)
                event_workers = event.get("workers")
                if isinstance(event_workers, int) and event_workers > worker_count:
                    now = time.time()
                    for _ in range(event_workers - worker_count):
                        workers.append(WorkerSlot(last_update=now))
                    worker_count = event_workers
            elif event_type == "start":
                label = event.get("label")
                if not label:
                    continue
                slot_idx = label_to_slot.get(label)
                if slot_idx is None:
                    slot_idx = _assign_slot(workers, label_to_slot, label, now)
                slot = workers[slot_idx]
                slot.status = "running"
                slot.dataset = str(event.get("dataset") or "-")
                slot.split = str(event.get("split") or "-")
                slot.day = str(event.get("day") or "-")
                slot.seed = event.get("seed")
                slot.generation = event.get("generation")
                slot.individual_index = event.get("individual_index")
                slot.start_time = event.get("timestamp", now)
                slot.last_update = slot.start_time
                slot.score = None
                slot.pnl = None
                slot.duration = 0.0
                slot.reason = "running"
                gen = slot.generation
                if gen is not None and gen not in generation_order:
                    generation_order.append(gen)
                    generation_order.sort()
                if gen is not None:
                    grid_status[(gen, slot_idx)] = "▶"
            elif event_type == "complete":
                label = event.get("label")
                if not label:
                    continue
                slot_idx = label_to_slot.get(label)
                if slot_idx is None:
                    slot_idx = _assign_slot(workers, label_to_slot, label, now)
                slot = workers[slot_idx]
                slot.status = "done"
                slot.dataset = str(event.get("dataset") or slot.dataset or "-")
                slot.split = str(event.get("split") or slot.split or "-")
                slot.day = str(event.get("day") or slot.day or "-")
                slot.seed = event.get("seed", slot.seed)
                if event.get("generation") is not None:
                    slot.generation = event.get("generation")
                slot.score = event.get("score")
                slot.pnl = event.get("pnl")
                slot.duration = event.get("duration", 0.0)
                slot.reason = event.get("reason", "done")
                slot.last_update = event.get("timestamp", now)
                gen = slot.generation
                if gen is not None and gen not in generation_order:
                    generation_order.append(gen)
                    generation_order.sort()
                if gen is not None:
                    success = slot.reason == "ok"
                    grid_status[(gen, slot_idx)] = "✓" if success else "✗"
            elif event_type == "generation":
                gen = event.get("generation")
                best = event.get("best_score")
                mean = event.get("mean_score")
                evals = event.get("evaluations_completed")
                if gen is not None and best is not None and mean is not None:
                    generation_text = (
                        f"Generation {gen}: best={best:.3f} mean={mean:.3f} evaluations={evals}"
                    )
                else:
                    generation_text = "Generation update received"
                if gen is not None and gen not in generation_order:
                    generation_order.append(gen)
                    generation_order.sort()
            elif event_type == "run_complete":
                run_completed = True
                duration = event.get("duration", 0.0)
                evaluations = event.get("evaluations_completed", 0)
                run_summary = f"Run completed in {duration:.1f}s with {evaluations} evaluations."

        now = time.time()
        for idx, slot in enumerate(workers):
            if slot.label and slot.status == "done" and now - slot.last_update > 5.0:
                label_to_slot.pop(slot.label, None)
                workers[idx].reset(now)

        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        _safe_addstr(stdscr, 0, 0, "MM CMA-ES TUI Monitor (press 'q' to exit)", curses.A_BOLD)
        _safe_addstr(stdscr, 1, 0, generation_text)
        status_line = run_summary if run_completed else (
            "Run status: active" if run_thread.is_alive() else "Run status: finished"
        )
        _safe_addstr(stdscr, 2, 0, status_line)
        _safe_addstr(stdscr, 3, 0, "-" * max(1, max_x - 1))

        grid_col_width = 4
        max_observed_gen = max(generation_order) if generation_order else 0
        max_gen = max(planned_generations, max_observed_gen)
        if max_gen <= 0:
            display_gens = [1]
        else:
            display_gens = list(range(1, max_gen + 1))

        legend_text = "Legend: ▶ running  ✓ success  ✗ failure  · pending"
        _safe_addstr(stdscr, 4, 0, legend_text)

        grid_prefix_width = 4  # width of "W00 "
        detail_min_width = 60
        available_width = max_x - grid_prefix_width - detail_min_width
        if available_width <= 0:
            max_cols = len(display_gens)
        else:
            max_cols = max(1, min(len(display_gens), available_width // grid_col_width))
        display_gens = display_gens[-max_cols:]

        header_cells = [f"G{gen:03d}".center(grid_col_width) for gen in display_gens]
        header = " " * grid_prefix_width + "|".join(header_cells)
        _safe_addstr(stdscr, 5, 0, header)
        _safe_addstr(stdscr, 6, 0, "-" * max(1, max_x - 1))
        base_row = 7

        rows_available = max(1, max_y - base_row - 4)
        display_workers = min(worker_count, rows_available)

        for idx in range(display_workers):
            slot = workers[idx] if idx < len(workers) else WorkerSlot(last_update=now)
            cells = []
            for gen in display_gens:
                ch = grid_status.get((gen, idx), "·")
                cells.append(f" {ch} ".center(grid_col_width))
            grid_line = "|".join(cells)

            label = slot.label or "<idle>"
            status = slot.status
            dataset = slot.dataset or "-"
            split = slot.split or "-"
            score = _format_score(slot.score)
            pnl = _format_pnl(slot.pnl)
            if slot.status == "running":
                duration_text = f"{now - slot.start_time:.1f}"
            else:
                duration_text = f"{slot.duration:.1f}" if slot.duration else "--"
            reason = slot.reason or ""

            prefix = f"W{idx:02} "
            detail = (
                f"{status:<8} {label:<20} {split:<6} {dataset:<16} score={score:<7} "
                f"pnl={pnl:<8} dur={duration_text:<6} {reason}"
            )
            max_detail_width = max_x - len(prefix) - len(grid_line) - 1
            if max_detail_width < 0:
                max_detail_width = 0
            detail = detail[:max_detail_width]
            line = f"{prefix}{grid_line} {detail}" if detail else f"{prefix}{grid_line}"
            _safe_addstr(stdscr, base_row + idx, 0, line)

        footer_row = base_row + display_workers + 2
        if run_completed and not run_thread.is_alive() and event_queue.empty():
            _safe_addstr(stdscr, footer_row, 0, "Run completed. Press 'q' to exit.")
        else:
            _safe_addstr(stdscr, footer_row, 0, "Press 'q' to exit once run completes.")

        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (ord('q'), ord('Q')):
            break

        time.sleep(0.05)

    run_thread.join()
