"""DearPyGui dashboard for monitoring CMA-ES optimisation runs."""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, List, Optional

from queue import Queue

from mmcore.events import (
    EvaluationStartEvent,
    EvaluationCompleteEvent,
    EvaluationResult,
    GenerationSummaryEvent,
    RunCompleteEvent,
    EventBus,
    drain_queue,
)

try:
    import dearpygui.dearpygui as dpg
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "DearPyGui is required for --gui runs. Install with `pip install dearpygui`."
    ) from exc


class GuiState:
    """Mutable state shared between the event handlers and the UI."""

    def __init__(self) -> None:
        self.evaluations: Dict[str, EvaluationResult] = {}
        self.running_labels: Dict[str, EvaluationStartEvent] = {}
        self.generation_numbers: List[int] = []
        self.best_scores: List[float] = []
        self.mean_scores: List[float] = []
        self.last_generation_event: Optional[GenerationSummaryEvent] = None
        self.run_completed: Optional[RunCompleteEvent] = None
        self.total_evals_seen: int = 0

    def upsert_start(self, event: EvaluationStartEvent) -> None:
        self.running_labels[event.label] = event

    def upsert_result(self, event: EvaluationCompleteEvent) -> None:
        result = event.result
        self.evaluations[result.label] = result
        self.running_labels.pop(result.label, None)
        self.total_evals_seen += 1

    def append_generation(self, event: GenerationSummaryEvent) -> None:
        self.last_generation_event = event
        self.generation_numbers.append(event.generation)
        self.best_scores.append(event.best_score)
        self.mean_scores.append(event.mean_score)

    def mark_complete(self, event: RunCompleteEvent) -> None:
        self.run_completed = event


class Dashboard:
    def __init__(self, event_queue: Queue, run_thread: threading.Thread) -> None:
        self.queue = event_queue
        self.run_thread = run_thread
        self.state = GuiState()
        self._plot_best_id = None
        self._plot_mean_id = None
        self._eval_table_id = "eval_table"
        self._status_text_id = None
        self._summary_id = None

    def setup(self) -> None:
        dpg.create_context()
        dpg.configure_app(docking=False)
        dpg.create_viewport(title="MM CMA-ES Monitor", width=1280, height=820)

        with dpg.window(tag="primary_window", label="Run Overview", pos=(10, 10), width=1240, height=120):
            self._status_text_id = dpg.add_text("Initialising optimisation...")
            self._summary_id = dpg.add_text("")

        with dpg.window(tag="scores_window", label="Generation Scores", pos=(660, 150), width=600, height=360):
            with dpg.plot(tag="scores_plot", label="Scores", height=-1, width=-1):
                dpg.add_plot_axis(dpg.mvXAxis, tag="scores_x_axis", label="Generation")
                y_axis = dpg.add_plot_axis(dpg.mvYAxis, tag="scores_y_axis", label="Score")
                self._plot_best_id = dpg.add_line_series([], [], label="Best", parent=y_axis)
                self._plot_mean_id = dpg.add_line_series([], [], label="Mean", parent=y_axis)
                try:
                    dpg.add_plot_legend(parent="scores_plot")
                except AttributeError:
                    try:
                        dpg.add_legend(parent="scores_plot")
                    except AttributeError:
                        pass

        with dpg.window(tag="eval_window", label="Evaluations", pos=(20, 150), width=620, height=640):
            with dpg.table(tag=self._eval_table_id, header_row=True, policy=dpg.mvTable_SizingStretchProp):
                for heading in [
                    "Label",
                    "Split",
                    "Dataset",
                    "Score",
                    "PnL",
                    "Inventory",
                    "Duration (s)",
                    "Reason",
                ]:
                    dpg.add_table_column(label=heading)

        dpg.setup_dearpygui()
        dpg.set_primary_window("primary_window", True)
        dpg.show_viewport()

    def _format_number(self, value: Optional[float]) -> str:
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            return "--"
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"{value/1_000:.2f}k"
        return f"{value:.3f}"

    def _update_table(self) -> None:
        for child in dpg.get_item_children(self._eval_table_id, 1) or []:
            dpg.delete_item(child)

        for start_evt in list(self.state.running_labels.values()):
            with dpg.table_row(parent=self._eval_table_id):
                dpg.add_text(start_evt.label)
                dpg.add_text(start_evt.split)
                dpg.add_text(start_evt.dataset_hint or "(pending)")
                dpg.add_text("...")
                dpg.add_text("...")
                dpg.add_text("...")
                elapsed = time.time() - start_evt.timestamp
                dpg.add_text(f"{elapsed:.1f}")
                dpg.add_text("running")

        for result in sorted(self.state.evaluations.values(), key=lambda r: (r.generation, r.label), reverse=True):
            with dpg.table_row(parent=self._eval_table_id):
                dpg.add_text(result.label)
                dpg.add_text(result.split)
                dpg.add_text(result.dataset or "--")
                dpg.add_text(self._format_number(result.score))
                dpg.add_text(self._format_number(result.pnl))
                dpg.add_text(self._format_number(result.inventory_abs))
                dpg.add_text(f"{result.duration:.1f}")
                dpg.add_text(result.reason)

    def _update_plots(self) -> None:
        if self.state.generation_numbers:
            dpg.set_value(self._plot_best_id, [self.state.generation_numbers, self.state.best_scores])
            dpg.set_value(self._plot_mean_id, [self.state.generation_numbers, self.state.mean_scores])
        else:
            dpg.set_value(self._plot_best_id, [[], []])
            dpg.set_value(self._plot_mean_id, [[], []])

    def _update_overview(self) -> None:
        if self.state.last_generation_event:
            evt = self.state.last_generation_event
            text = (
                f"Generation {evt.generation}: best={evt.best_score:.3f}, "
                f"mean={evt.mean_score:.3f}, evaluations={evt.evaluations_completed}"
            )
        else:
            text = "Awaiting first generation..."
        dpg.set_value(self._status_text_id, text)

        summary_lines = []
        if self.state.run_completed:
            rc = self.state.run_completed
            summary_lines.append(
                f"Run completed in {rc.duration:.1f}s with {rc.evaluations_completed} evaluations."
            )
        else:
            summary_lines.append(f"Evaluations observed: {self.state.total_evals_seen}")
            summary_lines.append("Run thread: " + ("active" if self.run_thread.is_alive() else "finished"))
        dpg.set_value(self._summary_id, "\n".join(summary_lines))

    def handle_events(self) -> None:
        for event in drain_queue(self.queue):
            if isinstance(event, EvaluationStartEvent):
                self.state.upsert_start(event)
            elif isinstance(event, EvaluationCompleteEvent):
                self.state.upsert_result(event)
            elif isinstance(event, GenerationSummaryEvent):
                self.state.append_generation(event)
            elif isinstance(event, RunCompleteEvent):
                self.state.mark_complete(event)

        self._update_table()
        self._update_plots()
        self._update_overview()

    def run(self) -> None:
        self.setup()

        def _render_callback(sender, app_data):  # pragma: no cover - GUI callback
            try:
                self.handle_events()
            except Exception as exc:
                print(f"[gui] render callback error: {exc}", flush=True)

        dpg.set_render_callback(_render_callback)
        print("[gui] DearPyGui viewport active", flush=True)
        dpg.start_dearpygui()
        dpg.destroy_context()


def launch_gui(event_bus: EventBus, run_thread: threading.Thread) -> None:
    queue = event_bus.subscribe()
    dashboard = Dashboard(queue, run_thread)
    dashboard.run()

