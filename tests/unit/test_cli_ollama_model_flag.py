"""Tests for --ollama-model CLI flag and updated --beam-size default."""
from __future__ import annotations
from click.testing import CliRunner
from unittest.mock import MagicMock, patch
from bn_en_translate.cli import main


def test_ollama_model_flag_sets_config() -> None:
    runner = CliRunner()
    captured_configs: list = []

    with runner.isolated_filesystem():
        with open("input.txt", "w", encoding="utf-8") as f:
            f.write("আমি ভাত খাই।")

        with patch("bn_en_translate.cli.get_translator") as mock_get, \
             patch("bn_en_translate.cli.TranslationPipeline"):

            mock_translator = MagicMock()
            mock_translator.__enter__ = lambda s: s
            mock_translator.__exit__ = MagicMock(return_value=False)

            def capture_config(config):
                captured_configs.append(config)
                return mock_translator

            mock_get.side_effect = capture_config

            result = runner.invoke(main, [
                "--input", "input.txt",
                "--output", "out.txt",
                "--ollama-model", "gemma3:12b",
            ])

    assert result.exit_code == 0, result.output
    assert len(captured_configs) == 1
    assert captured_configs[0].ollama_model == "gemma3:12b"


def test_ollama_model_flag_defaults_to_gemma3() -> None:
    runner = CliRunner()
    captured_configs: list = []

    with runner.isolated_filesystem():
        with open("input.txt", "w", encoding="utf-8") as f:
            f.write("আমি ভাত খাই।")

        with patch("bn_en_translate.cli.get_translator") as mock_get, \
             patch("bn_en_translate.cli.TranslationPipeline"):

            mock_translator = MagicMock()
            mock_translator.__enter__ = lambda s: s
            mock_translator.__exit__ = MagicMock(return_value=False)

            def capture_config(config):
                captured_configs.append(config)
                return mock_translator

            mock_get.side_effect = capture_config

            result = runner.invoke(main, [
                "--input", "input.txt",
                "--output", "out.txt",
            ])

    assert result.exit_code == 0, result.output
    assert captured_configs[0].ollama_model == "gemma3:12b"
