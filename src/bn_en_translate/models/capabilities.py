"""Declared translation-backend limits and optional feature support."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TranslatorCapabilities:
    """Immutable contract used to prepare requests for a translator backend.

    Limits apply to one translation request, including any model-specific special
    tokens.  A backend that cannot count tokens exactly must advertise that fact so
    callers can reserve a safety margin rather than silently exceeding its context.
    """

    max_input_tokens: int
    max_output_tokens: int
    supports_batching: bool
    supports_context_prompt: bool
    supports_glossary_constraints: bool
    supports_json_output: bool
    supports_seed: bool
    token_count_is_exact: bool
    supported_language_pairs: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.max_input_tokens <= 0 or self.max_output_tokens <= 0:
            raise ValueError("token limits must be positive")

    def supports_language_pair(self, source: str, target: str) -> bool:
        """Return whether a pair is explicitly supported (or unrestricted)."""
        return (
            not self.supported_language_pairs
            or (source, target) in self.supported_language_pairs
        )


CONSERVATIVE_CAPABILITIES = TranslatorCapabilities(
    max_input_tokens=384,
    max_output_tokens=128,
    supports_batching=False,
    supports_context_prompt=False,
    supports_glossary_constraints=False,
    supports_json_output=False,
    supports_seed=False,
    token_count_is_exact=False,
)


def _seq2seq(*, max_output_tokens: int = 512) -> TranslatorCapabilities:
    return TranslatorCapabilities(
        max_input_tokens=512,
        max_output_tokens=max_output_tokens,
        supports_batching=True,
        supports_context_prompt=False,
        supports_glossary_constraints=False,
        supports_json_output=False,
        supports_seed=False,
        token_count_is_exact=False,
    )


# Profiles are intentionally keyed by implementation class, not user-facing aliases:
# aliases may route to different HF/CT2 implementations depending on local assets.
ADAPTER_CAPABILITIES: dict[str, TranslatorCapabilities] = {
    name: _seq2seq()
    for name in (
        "NLLBTranslator", "NLLBCt2Translator", "IndicTrans2Translator",
        "IndicTrans2Ct2Translator", "SeamlessTranslator", "MiLMMTTranslator",
        "MiLMMT4BTranslator", "LMT60Translator", "SarvamTranslateTranslator",
        "KrutrimTranslateTranslator",
    )
}
ADAPTER_CAPABILITIES["MADLADTranslator"] = _seq2seq(max_output_tokens=256)
ADAPTER_CAPABILITIES["OllamaTranslator"] = TranslatorCapabilities(
    max_input_tokens=2048,
    max_output_tokens=1024,
    supports_batching=False,
    supports_context_prompt=True,
    supports_glossary_constraints=False,
    supports_json_output=False,
    supports_seed=False,
    token_count_is_exact=False,
)


def capabilities_for_adapter(adapter_name: str) -> TranslatorCapabilities:
    """Return the declared profile, or a conservative fallback for extensions."""
    return ADAPTER_CAPABILITIES.get(adapter_name, CONSERVATIVE_CAPABILITIES)
