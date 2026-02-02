"""Pipeline profiler. Lightweight per-stage timer for measuring latency."""

import time
from collections import defaultdict


class PipelineProfiler:
    def __init__(self, stages: list[str], report_interval: float = 5.0):
        self.stages = stages
        self.report_interval = report_interval
        self._starts: dict[str, float] = {}
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._last_report = time.time()
        self._frame_count = 0

    def start(self, stage: str) -> None:
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> None:
        if stage in self._starts:
            elapsed = time.perf_counter() - self._starts[stage]
            self._totals[stage] += elapsed
            self._counts[stage] += 1
            del self._starts[stage]

    def tick(self) -> None:
        """Call once per frame to track frame count."""
        self._frame_count += 1

    def report(self) -> str | None:
        """Return formatted report string every report_interval seconds, or None."""
        now = time.time()
        if now - self._last_report < self.report_interval:
            return None

        elapsed = now - self._last_report
        fps = self._frame_count / elapsed if elapsed > 0 else 0.0

        lines = [f"Pipeline profile ({self._frame_count} frames, {fps:.1f} FPS):"]
        total_ms = 0.0
        for stage in self.stages:
            count = self._counts[stage]
            if count == 0:
                lines.append(f"  {stage:>12s}: no data")
                continue
            avg_ms = (self._totals[stage] / count) * 1000
            total_ms += avg_ms
            lines.append(f"  {stage:>12s}: {avg_ms:6.2f} ms avg ({count} calls)")
        lines.append(f"  {'TOTAL':>12s}: {total_ms:6.2f} ms avg per frame")

        # Reset for next interval
        self._totals.clear()
        self._counts.clear()
        self._frame_count = 0
        self._last_report = now

        return "\n".join(lines)
