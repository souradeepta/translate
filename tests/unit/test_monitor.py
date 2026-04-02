"""Unit tests for ResourceMonitor — all hardware calls are mocked."""

from __future__ import annotations

import threading
import time
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    enabled: bool = True,
    interval: float = 0.1,
    backend: str = "none",
) -> object:
    from bn_en_translate.config import MonitorConfig
    from pathlib import Path
    return MonitorConfig(
        sample_interval_s=interval,
        enabled=enabled,
        db_path=Path("monitor/runs.db"),
        gpu_backend=backend,
    )


def _make_sample(**kwargs: float) -> object:
    from bn_en_translate.utils.monitor import ResourceSample
    defaults = dict(
        timestamp=1.0, cpu_pct=42.0, ram_mib=2048.0,
        swap_mib=0.0, gpu_vram_mib=1000.0, gpu_util_pct=75.0,
    )
    defaults.update(kwargs)
    return ResourceSample(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MonitorConfig validation
# ---------------------------------------------------------------------------

def test_monitor_config_defaults_valid() -> None:
    from bn_en_translate.config import MonitorConfig
    cfg = MonitorConfig()
    assert cfg.sample_interval_s > 0
    assert cfg.enabled is True
    assert cfg.gpu_backend in {"pynvml", "nvidia-smi", "none"}


def test_monitor_config_negative_interval_raises() -> None:
    from bn_en_translate.config import MonitorConfig
    with pytest.raises(ValueError, match="sample_interval_s"):
        MonitorConfig(sample_interval_s=-1.0)


def test_monitor_config_zero_interval_raises() -> None:
    from bn_en_translate.config import MonitorConfig
    with pytest.raises(ValueError, match="sample_interval_s"):
        MonitorConfig(sample_interval_s=0.0)


def test_monitor_config_invalid_backend_raises() -> None:
    from bn_en_translate.config import MonitorConfig
    with pytest.raises(ValueError, match="gpu_backend"):
        MonitorConfig(gpu_backend="cuda-direct")


# ---------------------------------------------------------------------------
# ResourceSample immutability
# ---------------------------------------------------------------------------

def test_resource_sample_is_frozen() -> None:
    sample = _make_sample()
    with pytest.raises(FrozenInstanceError):
        sample.cpu_pct = 99.0  # type: ignore[misc]


def test_resource_sample_fields_accessible() -> None:
    from bn_en_translate.utils.monitor import ResourceSample
    s = ResourceSample(
        timestamp=5.0, cpu_pct=50.0, ram_mib=1024.0,
        swap_mib=0.0, gpu_vram_mib=500.0, gpu_util_pct=60.0,
    )
    assert s.cpu_pct == 50.0
    assert s.ram_mib == 1024.0


# ---------------------------------------------------------------------------
# ResourceSummary aggregation
# ---------------------------------------------------------------------------

def test_resource_summary_from_empty_samples_returns_zeros() -> None:
    from bn_en_translate.utils.monitor import ResourceSummary
    s = ResourceSummary.from_samples([], duration_s=5.0, disk_read_mb=1.0, disk_write_mb=0.5)
    assert s.sample_count == 0
    assert s.cpu_peak_pct == 0.0
    assert s.ram_peak_mib == 0.0
    assert s.disk_read_mb == 1.0
    assert s.duration_s == 5.0


def test_resource_summary_from_samples_computes_correct_peak() -> None:
    from bn_en_translate.utils.monitor import ResourceSummary
    samples = [
        _make_sample(cpu_pct=20.0, ram_mib=1000.0, gpu_util_pct=30.0),
        _make_sample(cpu_pct=80.0, ram_mib=2000.0, gpu_util_pct=90.0),
        _make_sample(cpu_pct=50.0, ram_mib=1500.0, gpu_util_pct=60.0),
    ]
    s = ResourceSummary.from_samples(samples, duration_s=3.0, disk_read_mb=0.0, disk_write_mb=0.0)  # type: ignore[arg-type]
    assert s.cpu_peak_pct == 80.0
    assert s.ram_peak_mib == 2000.0
    assert s.gpu_util_peak_pct == 90.0


def test_resource_summary_from_samples_computes_correct_avg() -> None:
    from bn_en_translate.utils.monitor import ResourceSummary
    samples = [
        _make_sample(cpu_pct=10.0, ram_mib=1000.0),
        _make_sample(cpu_pct=30.0, ram_mib=3000.0),
    ]
    s = ResourceSummary.from_samples(samples, duration_s=2.0, disk_read_mb=0.0, disk_write_mb=0.0)  # type: ignore[arg-type]
    assert s.cpu_avg_pct == pytest.approx(20.0)
    assert s.ram_avg_mib == pytest.approx(2000.0)


def test_resource_summary_skips_unavailable_gpu_util() -> None:
    from bn_en_translate.utils.monitor import ResourceSummary
    # gpu_util_pct=-1.0 signals unavailable — should be excluded from avg/peak
    samples = [
        _make_sample(gpu_util_pct=-1.0),
        _make_sample(gpu_util_pct=-1.0),
    ]
    s = ResourceSummary.from_samples(samples, duration_s=2.0, disk_read_mb=0.0, disk_write_mb=0.0)  # type: ignore[arg-type]
    assert s.gpu_util_peak_pct == 0.0
    assert s.gpu_util_avg_pct == 0.0


# ---------------------------------------------------------------------------
# ResourceMonitor basic behaviour
# ---------------------------------------------------------------------------

def test_resource_monitor_enter_returns_self() -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor
    cfg = _make_config()
    m = ResourceMonitor(config=cfg)
    with patch("psutil.cpu_percent"), patch("psutil.disk_io_counters", return_value=None):
        result = m.__enter__()
    m._stop_event.set()
    if m._thread:
        m._thread.join(timeout=1.0)
    assert result is m


def test_resource_monitor_generates_run_id_if_none() -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor
    m = ResourceMonitor(config=_make_config())
    assert len(m.run_id) == 32  # uuid4 hex


def test_resource_monitor_accepts_provided_run_id() -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor
    m = ResourceMonitor(config=_make_config(), run_id="fixed-id-123")
    assert m.run_id == "fixed-id-123"


def test_resource_monitor_summary_none_before_exit() -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor
    m = ResourceMonitor(config=_make_config())
    assert m.summary is None


def test_resource_monitor_disabled_skips_all_work() -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor
    cfg = _make_config(enabled=False)
    m = ResourceMonitor(config=cfg)
    with m:
        pass
    assert m.summary is None
    assert m.samples == []
    assert m._thread is None


def test_resource_monitor_exit_sets_summary() -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor
    cfg = _make_config(interval=0.05)

    mock_io = MagicMock()
    mock_io.read_bytes = 0
    mock_io.write_bytes = 0

    with (
        patch("psutil.cpu_percent", return_value=50.0),
        patch("psutil.disk_io_counters", return_value=mock_io),
        patch("psutil.virtual_memory", return_value=MagicMock(used=2 * 1024**3)),
        patch("psutil.swap_memory", return_value=MagicMock(used=0)),
    ):
        with ResourceMonitor(config=cfg) as m:
            time.sleep(0.15)  # let the thread take at least one sample

    assert m.summary is not None
    assert m.summary.duration_s > 0


def test_resource_monitor_stop_event_signals_thread_exit() -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor
    cfg = _make_config(interval=0.05)

    mock_io = MagicMock()
    mock_io.read_bytes = 0
    mock_io.write_bytes = 0

    with (
        patch("psutil.cpu_percent", return_value=10.0),
        patch("psutil.disk_io_counters", return_value=mock_io),
        patch("psutil.virtual_memory", return_value=MagicMock(used=1 * 1024**3)),
        patch("psutil.swap_memory", return_value=MagicMock(used=0)),
    ):
        m = ResourceMonitor(config=cfg)
        m.__enter__()
        assert m._thread is not None
        assert m._thread.is_alive()
        m.__exit__(None, None, None)

    assert not m._thread.is_alive()


def test_resource_monitor_exit_joins_thread(mocker: object) -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor

    mock_thread = mocker.MagicMock()  # type: ignore[union-attr]
    mock_thread.is_alive.return_value = False

    mock_io = MagicMock()
    mock_io.read_bytes = 0
    mock_io.write_bytes = 0

    with (
        patch("threading.Thread", return_value=mock_thread),
        patch("psutil.cpu_percent"),
        patch("psutil.disk_io_counters", return_value=mock_io),
    ):
        m = ResourceMonitor(config=_make_config())
        m.__enter__()
        m.__exit__(None, None, None)

    mock_thread.join.assert_called_once()


# ---------------------------------------------------------------------------
# GPU backend
# ---------------------------------------------------------------------------

def test_get_gpu_stats_backend_none_returns_unavailable() -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor
    m = ResourceMonitor(config=_make_config(backend="none"))
    vram, util = m._get_gpu_stats()
    assert vram == 0.0
    assert util == -1.0


def test_get_gpu_stats_pynvml_returns_vram_and_util(mocker: object) -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor

    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.return_value = None
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(used=2 * 1024**3)
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=67)
    mocker.patch.dict("sys.modules", {"pynvml": mock_pynvml})  # type: ignore[union-attr]

    m = ResourceMonitor(config=_make_config(backend="pynvml"))
    # Manually init nvml as __enter__ would
    m._nvml_available = m._init_nvml()
    vram, util = m._get_gpu_stats()
    assert vram == pytest.approx(2048.0)
    assert util == pytest.approx(67.0)


def test_get_gpu_stats_pynvml_unavailable_falls_back_to_nvidia_smi(mocker: object) -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor

    # pynvml raises on init
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.side_effect = Exception("no nvml")
    mocker.patch.dict("sys.modules", {"pynvml": mock_pynvml})  # type: ignore[union-attr]

    m = ResourceMonitor(config=_make_config(backend="pynvml"))
    m._nvml_available = False  # pynvml failed

    with patch.object(m, "_get_gpu_stats_nvidia_smi", return_value=(1500.0, 45.0)):
        vram, util = m._get_gpu_stats()

    assert vram == pytest.approx(1500.0)
    assert util == pytest.approx(45.0)


def test_get_gpu_stats_nvidia_smi_parse_error_returns_zeros(mocker: object) -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor
    import subprocess

    m = ResourceMonitor(config=_make_config(backend="nvidia-smi"))
    with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
        vram, util = m._get_gpu_stats_nvidia_smi()

    assert vram == 0.0
    assert util == -1.0


# ---------------------------------------------------------------------------
# take_sample integration
# ---------------------------------------------------------------------------

def test_take_sample_uses_psutil_cpu_percent() -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor
    cfg = _make_config()
    m = ResourceMonitor(config=cfg)

    with (
        patch("psutil.cpu_percent", return_value=77.0) as mock_cpu,
        patch("psutil.virtual_memory", return_value=MagicMock(used=1 * 1024**3)),
        patch("psutil.swap_memory", return_value=MagicMock(used=0)),
    ):
        sample = m._take_sample()

    mock_cpu.assert_called_once_with(interval=None)
    assert sample.cpu_pct == 77.0


def test_take_sample_timestamp_increases() -> None:
    from bn_en_translate.utils.monitor import ResourceMonitor
    m = ResourceMonitor(config=_make_config())
    with (
        patch("psutil.cpu_percent", return_value=10.0),
        patch("psutil.virtual_memory", return_value=MagicMock(used=1 * 1024**3)),
        patch("psutil.swap_memory", return_value=MagicMock(used=0)),
    ):
        s1 = m._take_sample()
        time.sleep(0.01)
        s2 = m._take_sample()
    assert s2.timestamp >= s1.timestamp


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

def test_format_summary_returns_string() -> None:
    from bn_en_translate.utils.monitor import ResourceSummary, format_summary
    s = ResourceSummary.from_samples([], duration_s=10.0, disk_read_mb=5.0, disk_write_mb=2.0)
    result = format_summary(s)
    assert isinstance(result, str)
    assert "duration=10.0s" in result
