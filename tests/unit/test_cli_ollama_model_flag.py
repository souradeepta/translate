"""Tests for --ollama-model CLI flag and updated --beam-size default."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from bn_en_translate.cli import main
from bn_en_translate.models.factory import supported_model_names


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


def test_batch_size_flag_sets_chunk_config() -> None:
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
            mock_get.side_effect = lambda config: captured_configs.append(config) or mock_translator

            result = runner.invoke(
                main,
                ["--input", "input.txt", "--output", "out.txt", "--batch-size", "3"],
            )

    assert result.exit_code == 0, result.output
    assert captured_configs[0].chunk.batch_size == 3


def test_batch_size_flag_rejects_zero() -> None:
    result = CliRunner().invoke(
        main,
        ["--input", "input.txt", "--output", "out.txt", "--batch-size", "0"],
    )

    assert result.exit_code != 0
    assert "Invalid value for '--batch-size'" in result.output


def test_cli_help_lists_every_registered_model() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    for name in supported_model_names():
        assert name in result.output
