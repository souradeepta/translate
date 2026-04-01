"""Shared pytest fixtures and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from bn_en_translate.models.base import TranslatorBase
from bn_en_translate.config import PipelineConfig


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class MockTranslator(TranslatorBase):
    """
    Test double that prefixes each input with '[MOCK] '.
    Allows verifying reassembly order, batching, and pipeline flow
    without downloading any model.
    """

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        return [f"[MOCK] {t}" for t in texts]


@pytest.fixture
def mock_translator() -> MockTranslator:
    return MockTranslator()


@pytest.fixture
def default_config() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def short_bengali_text() -> str:
    return (FIXTURES_DIR / "sample_short.bn.txt").read_text(encoding="utf-8")


@pytest.fixture
def medium_bengali_text() -> str:
    return (FIXTURES_DIR / "sample_medium.bn.txt").read_text(encoding="utf-8")
