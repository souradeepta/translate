"""TDD tests for CUDA check utilities (all mocked — no GPU required)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from bn_en_translate.utils import cuda_check
from bn_en_translate.utils.cuda_check import (
    get_best_device,
    get_free_vram_mib,
    get_total_vram_mib,
    is_cuda_available,
)


def test_returns_false_when_torch_not_installed() -> None:
    with patch.dict("sys.modules", {"torch": None}):
        # Re-import to pick up the patched modules
        import importlib

        import bn_en_translate.utils.cuda_check as m

        importlib.reload(m)
        # When torch is unavailable, should return False gracefully
        result = m.is_cuda_available()
        assert result is False
        importlib.reload(m)  # restore


def test_returns_false_when_cuda_not_available() -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = is_cuda_available()
        assert result is False


def test_returns_true_when_cuda_available() -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = is_cuda_available()
        assert result is True


def test_get_best_device_returns_cuda_when_available() -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": mock_torch}):
        assert get_best_device() == "cuda"


def test_get_best_device_returns_cpu_when_unavailable() -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        assert get_best_device() == "cpu"


def test_get_free_vram_returns_zero_when_no_cuda() -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        assert get_free_vram_mib() == 0


def test_get_free_vram_returns_correct_mib() -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    # 4 GB free, 8 GB total
    mock_torch.cuda.mem_get_info.return_value = (4 * 1024**3, 8 * 1024**3)
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = get_free_vram_mib()
        assert result == 4096


def test_get_total_vram_returns_correct_mib() -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    props = MagicMock()
    props.total_memory = 8 * 1024**3
    mock_torch.cuda.get_device_properties.return_value = props
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = get_total_vram_mib()
        assert result == 8192


def test_reset_cuda_state_no_torch_is_noop() -> None:
    """Must never raise when torch is not installed (CI, CPU-only envs)."""
    with patch.dict("sys.modules", {"torch": None}):
        import importlib

        import bn_en_translate.utils.cuda_check as m

        importlib.reload(m)
        m.reset_cuda_state()  # should not raise
        importlib.reload(m)  # restore


def test_reset_cuda_state_no_cuda_is_noop(monkeypatch) -> None:
    """No-op (no calls, no raise) when CUDA is unavailable."""
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    cuda_check.reset_cuda_state()

    fake_torch.cuda.empty_cache.assert_not_called()
    fake_torch.cuda.reset_peak_memory_stats.assert_not_called()


def test_reset_cuda_state_calls_torch(monkeypatch) -> None:
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    cuda_check.reset_cuda_state()

    fake_torch.cuda.empty_cache.assert_called_once()
    fake_torch.cuda.reset_peak_memory_stats.assert_called_once()


def test_ensure_vram_available_raises_when_insufficient(monkeypatch) -> None:
    monkeypatch.setattr(cuda_check, "get_free_vram_mib", lambda: 1000)
    with pytest.raises(RuntimeError, match="polish pass"):
        cuda_check.ensure_vram_available(4800, context="Ollama polish pass")


def test_ensure_vram_available_passes_when_sufficient(monkeypatch) -> None:
    monkeypatch.setattr(cuda_check, "get_free_vram_mib", lambda: 6000)
    cuda_check.ensure_vram_available(4800, context="Ollama polish pass")  # no raise
