"""Contract tests for the redistributable document-quality fixture."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "book"


def _blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return text.split("\n\n")


def test_fixture_provenance_is_explicit_and_redistributable() -> None:
    metadata = json.loads((FIXTURE_DIR / "expected_terms.json").read_text(encoding="utf-8"))
    provenance = metadata["provenance"]

    assert provenance["redistributable"] is True
    assert provenance["license"] == "CC0-1.0"
    assert provenance["source_language"] == "bn"
    assert provenance["target_language"] == "en"


def test_source_and_reference_block_counts_match() -> None:
    source = _blocks(FIXTURE_DIR / "consistency_source.bn.txt")
    reference = _blocks(FIXTURE_DIR / "consistency_reference.en.txt")
    metadata = json.loads((FIXTURE_DIR / "expected_terms.json").read_text(encoding="utf-8"))

    assert len(source) == len(reference) == metadata["blocks"]


def test_expected_terms_reference_valid_source_blocks() -> None:
    source = _blocks(FIXTURE_DIR / "consistency_source.bn.txt")
    metadata = json.loads((FIXTURE_DIR / "expected_terms.json").read_text(encoding="utf-8"))

    for term in metadata["terms"]:
        for block_number in term["occurrences"]:
            assert 1 <= block_number <= len(source)
            assert term["source"] in source[block_number - 1]
