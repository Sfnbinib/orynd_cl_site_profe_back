"""Macro layer — natural language → CoreOps.

Founder ask: *"GPT-обученные макросы, в реал-тайме нейронка поняла команду
и создала прямоугольник, выдавила, развернула, добавила отверстие"*.

Phase 0 (here): regex/keyword parser — закрывает basic patterns без LLM.
Phase 1 (later): добавим Anthropic call для arbitrary text → CoreOps
через model_router. Этот файл остаётся entry point.
"""

from orynd_core.services.macro.parser import (
    MacroParseResult,
    parse_text_to_coreops,
)

__all__ = ["MacroParseResult", "parse_text_to_coreops"]
