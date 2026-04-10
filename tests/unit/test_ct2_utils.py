"""Unit tests for shared CTranslate2 utilities."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pytest


def test_probe_compute_type_cpu_returns_int8() -> None:
    """CPU path short-circuits without loading ctranslate2."""
    from bn_en_translate.utils.ct2_utils import probe_compute_type
    result = probe_compute_type("/fake/model", "cpu", lambda t: None)
    assert result == "int8"


def test_probe_compute_type_returns_first_working_type() -> None:
    from bn_en_translate.utils.ct2_utils import probe_compute_type

    mock_translator = MagicMock()
    probe_fn = MagicMock()

    mock_ct2 = MagicMock()
    mock_ct2.get_supported_compute_types.return_value = {"int8_float16", "float16", "float32"}
    mock_ct2.Translator.return_value = mock_translator

    with patch.dict("sys.modules", {"ctranslate2": mock_ct2}):
        result = probe_compute_type("/fake/model", "cuda", probe_fn)

    assert result == "int8_float16"
    probe_fn.assert_called_once_with(mock_translator)


def test_probe_compute_type_skips_failed_type() -> None:
    """If int8_float16 probe raises, falls through to the next candidate."""
    from bn_en_translate.utils.ct2_utils import probe_compute_type

    bad_translator = MagicMock()
    good_translator = MagicMock()

    probe_call_count = [0]

    def probe_fn(t: object) -> None:
        probe_call_count[0] += 1
        if t is bad_translator:
            raise RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED")
        # good_translator passes without raising

    mock_ct2 = MagicMock()
    mock_ct2.get_supported_compute_types.return_value = {"int8_float16", "float16"}
    mock_ct2.Translator.side_effect = [bad_translator, good_translator]

    with patch.dict("sys.modules", {"ctranslate2": mock_ct2}):
        result = probe_compute_type("/fake/model", "cuda", probe_fn)

    assert result == "float16"
    assert probe_call_count[0] == 2


def test_probe_compute_type_fallback_to_float32() -> None:
    """Returns float32 if all candidates in _supported_ fail the probe."""
    from bn_en_translate.utils.ct2_utils import probe_compute_type

    mock_ct2 = MagicMock()
    mock_ct2.get_supported_compute_types.return_value = {"float32"}
    failing_translator = MagicMock()
    mock_ct2.Translator.return_value = failing_translator

    def always_fail(t: object) -> None:
        raise RuntimeError("always fails")

    with patch.dict("sys.modules", {"ctranslate2": mock_ct2}):
        result = probe_compute_type("/fake/model", "cuda", always_fail)

    assert result == "float32"


def test_probe_compute_type_skips_unsupported_types() -> None:
    """Only tries compute types reported as supported by the device."""
    from bn_en_translate.utils.ct2_utils import probe_compute_type

    mock_translator = MagicMock()
    probe_fn = MagicMock()

    mock_ct2 = MagicMock()
    # Only float32 is reported as supported — int8_float16, int8, float16, bfloat16 all absent
    mock_ct2.get_supported_compute_types.return_value = {"float32"}
    mock_ct2.Translator.return_value = mock_translator

    with patch.dict("sys.modules", {"ctranslate2": mock_ct2}):
        result = probe_compute_type("/fake/model", "cuda", probe_fn)

    assert result == "float32"
    assert mock_ct2.Translator.call_count == 1
