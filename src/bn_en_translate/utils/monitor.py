"""Background resource monitor — CPU, RAM, swap, disk, GPU.

Usage::

    from bn_en_translate.config import MonitorConfig
    from bn_en_translate.utils.monitor import ResourceMonitor

    with ResourceMonitor() as m:
        do_expensive_work()

    summary = m.summary  # ResourceSummary with peak/avg stats
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Immutable sample — one snapshot per tick
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ResourceSample:
    """One point-in-time hardware snapshot taken by the background thread."""

    timestamp: float       # time.monotonic()
    cpu_pct: float         # 0–100
    ram_mib: float         # used RAM in MiB
    swap_mib: float        # used swap in MiB
    gpu_vram_mib: float    # used VRAM in MiB; 0.0 if unavailable
    gpu_util_pct: float    # 0–100; -1.0 signals "unavailable"


# ---------------------------------------------------------------------------
# Aggregated summary computed at run end
# ---------------------------------------------------------------------------

@dataclass
class ResourceSummary:
    """Aggregated stats from all ResourceSamples taken during a run."""

    sample_count: int
    duration_s: float

    cpu_peak_pct: float
    cpu_avg_pct: float

    ram_peak_mib: float
    ram_avg_mib: float

    swap_peak_mib: float
    swap_avg_mib: float

    disk_read_mb: float    # total MB read during the run (driver delta)
    disk_write_mb: float   # total MB written during the run (driver delta)

    gpu_vram_peak_mib: float
    gpu_vram_avg_mib: float
    gpu_util_peak_pct: float
    gpu_util_avg_pct: float

    @classmethod
    def from_samples(
        cls,
        samples: list[ResourceSample],
        duration_s: float,
        disk_read_mb: float,
        disk_write_mb: float,
    ) -> ResourceSummary:
        """Aggregate a list of samples. Returns zero-valued summary for empty list."""
        if not samples:
            return cls(
                sample_count=0,
                duration_s=duration_s,
                cpu_peak_pct=0.0,
                cpu_avg_pct=0.0,
                ram_peak_mib=0.0,
                ram_avg_mib=0.0,
                swap_peak_mib=0.0,
                swap_avg_mib=0.0,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                gpu_vram_peak_mib=0.0,
                gpu_vram_avg_mib=0.0,
                gpu_util_peak_pct=0.0,
                gpu_util_avg_pct=0.0,
            )

        n = len(samples)
        cpu_vals = [s.cpu_pct for s in samples]
        ram_vals = [s.ram_mib for s in samples]
        swap_vals = [s.swap_mib for s in samples]
        vram_vals = [s.gpu_vram_mib for s in samples]
        util_vals = [s.gpu_util_pct for s in samples if s.gpu_util_pct >= 0]

        return cls(
            sample_count=n,
            duration_s=duration_s,
            cpu_peak_pct=max(cpu_vals),
            cpu_avg_pct=sum(cpu_vals) / n,
            ram_peak_mib=max(ram_vals),
            ram_avg_mib=sum(ram_vals) / n,
            swap_peak_mib=max(swap_vals),
            swap_avg_mib=sum(swap_vals) / n,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            gpu_vram_peak_mib=max(vram_vals),
            gpu_vram_avg_mib=sum(vram_vals) / n,
            gpu_util_peak_pct=max(util_vals) if util_vals else 0.0,
            gpu_util_avg_pct=sum(util_vals) / len(util_vals) if util_vals else 0.0,
        )


# ---------------------------------------------------------------------------
# Monitor context manager
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Context manager that samples CPU/RAM/Swap/GPU at a fixed interval.

    GPU stats are queried via pynvml (preferred) → nvidia-smi subprocess → zeros.
    The background sampling thread is a daemon thread so it dies with the process
    if __exit__ is never called.

    After ``__exit__``, the ``summary`` property contains aggregated stats.
    The ``run_id`` is a UUID4 hex string, stable across runs.
    """

    def __init__(
        self,
        config: object | None = None,
        run_id: str | None = None,
    ) -> None:
        # Import here to allow config=None convenience
        if config is None:
            from bn_en_translate.config import MonitorConfig
            config = MonitorConfig()

        self._config = config
        self._run_id: str = run_id if run_id is not None else uuid4().hex
        self._samples: list[ResourceSample] = []
        self._summary: ResourceSummary | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._started_at: datetime | None = None

        # Disk I/O snapshots (captured in __enter__ / __exit__)
        self._disk_read_start: float = 0.0
        self._disk_write_start: float = 0.0

        # pynvml handle — initialised once in __enter__ if backend == 'pynvml'
        self._nvml_handle: object | None = None
        self._nvml_available: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def started_at(self) -> datetime | None:
        return self._started_at

    @property
    def summary(self) -> ResourceSummary | None:
        """Available only after __exit__. None if disabled or still running."""
        return self._summary

    @property
    def samples(self) -> list[ResourceSample]:
        return list(self._samples)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ResourceMonitor:
        if not self._config.enabled:  # type: ignore[union-attr]
            return self

        self._started_at = datetime.now(tz=timezone.utc)
        self._start_time = time.monotonic()

        # Capture disk I/O baseline
        try:
            import psutil
            io = psutil.disk_io_counters()
            if io is not None:
                self._disk_read_start = io.read_bytes
                self._disk_write_start = io.write_bytes
        except Exception:
            pass

        # Initialise GPU backend once
        self._nvml_available = self._init_nvml()

        # Prime CPU measurement — first call returns 0.0 (measures since last call)
        try:
            import psutil
            psutil.cpu_percent(interval=0.1)
        except Exception:
            pass

        # Start background sampling thread
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sampling_loop,
            daemon=True,
            name="resource-monitor",
        )
        self._thread.start()
        logger.debug("ResourceMonitor started (run_id=%s)", self._run_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if not self._config.enabled:  # type: ignore[union-attr]
            return

        # Signal and join sampling thread
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        duration_s = time.monotonic() - self._start_time

        # Capture disk I/O delta
        disk_read_mb = disk_write_mb = 0.0
        try:
            import psutil
            io = psutil.disk_io_counters()
            if io is not None:
                disk_read_mb = (io.read_bytes - self._disk_read_start) / 1024**2
                disk_write_mb = (io.write_bytes - self._disk_write_start) / 1024**2
        except Exception:
            pass

        # Shutdown NVML
        self._shutdown_nvml()

        self._summary = ResourceSummary.from_samples(
            self._samples,
            duration_s=duration_s,
            disk_read_mb=max(disk_read_mb, 0.0),
            disk_write_mb=max(disk_write_mb, 0.0),
        )
        logger.debug(
            "ResourceMonitor stopped (run_id=%s, samples=%d, duration=%.1fs)",
            self._run_id, len(self._samples), duration_s,
        )

    # ------------------------------------------------------------------
    # Sampling loop (runs in daemon thread)
    # ------------------------------------------------------------------

    def _sampling_loop(self) -> None:
        interval = self._config.sample_interval_s  # type: ignore[union-attr]
        while not self._stop_event.wait(timeout=interval):
            try:
                sample = self._take_sample()
                self._samples.append(sample)
            except Exception as exc:
                logger.debug("Monitor sample failed: %s", exc)

    def _take_sample(self) -> ResourceSample:
        """Capture one snapshot of system resources."""
        import psutil

        cpu_pct = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        ram_mib = vm.used / 1024**2
        swap_mib = sw.used / 1024**2

        gpu_vram_mib, gpu_util_pct = self._get_gpu_stats()

        return ResourceSample(
            timestamp=time.monotonic(),
            cpu_pct=cpu_pct,
            ram_mib=ram_mib,
            swap_mib=swap_mib,
            gpu_vram_mib=gpu_vram_mib,
            gpu_util_pct=gpu_util_pct,
        )

    # ------------------------------------------------------------------
    # GPU backend
    # ------------------------------------------------------------------

    def _init_nvml(self) -> bool:
        """Initialise pynvml. Returns True on success."""
        backend = self._config.gpu_backend  # type: ignore[union-attr]
        if backend != "pynvml":
            return False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return True
        except Exception as exc:
            logger.debug("pynvml init failed (%s); falling back to nvidia-smi", exc)
            return False

    def _shutdown_nvml(self) -> None:
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml_handle = None
        self._nvml_available = False

    def _get_gpu_stats(self) -> tuple[float, float]:
        """Return (vram_used_mib, util_pct). Returns (0.0, -1.0) if unavailable."""
        backend = self._config.gpu_backend  # type: ignore[union-attr]

        if backend == "none":
            return 0.0, -1.0

        if backend == "pynvml" and self._nvml_available and self._nvml_handle is not None:
            try:
                import pynvml
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                return mem.used / 1024**2, float(util.gpu)
            except Exception as exc:
                logger.debug("pynvml sample failed: %s", exc)
                return 0.0, -1.0

        if backend in ("pynvml", "nvidia-smi"):
            # pynvml unavailable or failed — try nvidia-smi subprocess
            return self._get_gpu_stats_nvidia_smi()

        return 0.0, -1.0

    def _get_gpu_stats_nvidia_smi(self) -> tuple[float, float]:
        """Parse GPU stats from nvidia-smi. Returns (0.0, -1.0) on any error."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                timeout=3,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            parts = out.split(",")
            vram_mib = float(parts[0].strip())
            util_pct = float(parts[1].strip())
            return vram_mib, util_pct
        except Exception as exc:
            logger.debug("nvidia-smi parse failed: %s", exc)
            return 0.0, -1.0


# ---------------------------------------------------------------------------
# Convenience: human-readable summary line
# ---------------------------------------------------------------------------

def format_summary(summary: ResourceSummary) -> str:
    """Return a single-line human-readable summary suitable for logging."""
    return (
        f"duration={summary.duration_s:.1f}s | "
        f"cpu_peak={summary.cpu_peak_pct:.0f}% avg={summary.cpu_avg_pct:.0f}% | "
        f"ram_peak={summary.ram_peak_mib:.0f} MiB | "
        f"swap={summary.swap_peak_mib:.0f} MiB | "
        f"vram_peak={summary.gpu_vram_peak_mib:.0f} MiB "
        f"gpu_util_peak={summary.gpu_util_peak_pct:.0f}% | "
        f"disk r={summary.disk_read_mb:.1f} MB w={summary.disk_write_mb:.1f} MB"
    )
