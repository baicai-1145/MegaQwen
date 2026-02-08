from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class StageSummary:
    cpu_ms: dict[str, float]
    cuda_ms: dict[str, float]

    @property
    def total_cpu_ms(self) -> float:
        return float(sum(self.cpu_ms.values()))

    @property
    def total_cuda_ms(self) -> float:
        return float(sum(self.cuda_ms.values()))


class StageTimer:
    """Tiny profiler for coarse stage timing.

    - CPU stages use wall-clock (perf_counter).
    - CUDA stages use events; results require synchronize() at summary time.

    Intended for debug/profiling only (not enabled by default).
    """

    def __init__(self, *, enabled: bool = True):
        self.enabled = bool(enabled)
        self._cpu: list[tuple[str, float]] = []
        self._cuda: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def reset(self) -> None:
        self._cpu.clear()
        self._cuda.clear()

    @contextmanager
    def cpu(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._cpu.append((name, time.perf_counter() - t0))

    @contextmanager
    def cuda(self, name: str):
        if not self.enabled or not torch.cuda.is_available():
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._cuda.append((name, start, end))

    def summary(self) -> StageSummary:
        cpu_ms: dict[str, float] = {}
        for name, dt_s in self._cpu:
            cpu_ms[name] = cpu_ms.get(name, 0.0) + dt_s * 1000.0

        cuda_ms: dict[str, float] = {}
        if self._cuda:
            torch.cuda.synchronize()
            for name, start, end in self._cuda:
                cuda_ms[name] = cuda_ms.get(name, 0.0) + float(start.elapsed_time(end))

        return StageSummary(cpu_ms=cpu_ms, cuda_ms=cuda_ms)

    def format(self) -> str:
        s = self.summary()
        lines: list[str] = []
        if s.cpu_ms:
            lines.append("CPU stages (ms):")
            for k in sorted(s.cpu_ms.keys()):
                lines.append(f"  {k}: {s.cpu_ms[k]:.3f}")
            lines.append(f"  total_cpu: {s.total_cpu_ms:.3f}")
        if s.cuda_ms:
            lines.append("CUDA stages (ms):")
            for k in sorted(s.cuda_ms.keys()):
                lines.append(f"  {k}: {s.cuda_ms[k]:.3f}")
            lines.append(f"  total_cuda: {s.total_cuda_ms:.3f}")
        return "\n".join(lines)

