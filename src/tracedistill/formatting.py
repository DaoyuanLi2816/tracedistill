"""The format contract for reasoning-trace distillation.

The single most fragile, highest-leverage piece of trace distillation is the SFT
*target format*. It must be byte-for-byte identical to the eval / inference protocol
so the trained model reliably emits a parseable final answer. This module builds that
target:

    <think>                    <- added by the chat template (enable_thinking=True)
    (reasoning trace, with any upstream \\boxed{...} stripped)
    </think>                   <- added here, by build_record()
    \\boxed{official answer}    <- added here, by build_record()

The key idea is to **decouple the reasoning from the answer**: the chain-of-thought
comes from an upstream teacher trace, but the final ``\\boxed{}`` is rewritten with the
*authoritative* label, so the student learns the *procedure* while being anchored to
the *correct* answer. The user turn is suffixed with the exact instruction the grader
appends at eval time, so "training input ≈ eval input".

This module is dependency-light (only the standard library) so it can be imported and
unit-tested without torch/transformers.

Example:
    >>> rec = build_record("What is 2+2?", "We add two and two. \\boxed{5}", 4)
    >>> rec["messages"][0]["content"].endswith("\\boxed{your answer}`")
    True
    >>> rec["messages"][1]["content"]
    'We add two and two.\\n</think>\\n\\\\boxed{4}'
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

__all__ = [
    "DEFAULT_PROMPT_SUFFIX",
    "strip_boxed",
    "build_record",
    "build_records",
]

#: The instruction the competition grader appends to every prompt. Training reuses it
#: verbatim so the training distribution matches the eval distribution.
DEFAULT_PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

_BOXED_START_RE = re.compile(r"\\boxed\{")


def strip_boxed(text: str) -> str:
    """Remove every ``\\boxed{...}`` span from *text* (used to drop the upstream trace's
    own final answer before re-attaching the authoritative one).

    Tracks brace depth rather than matching up to the first ``}``, so nested braces
    inside the boxed content (``\\boxed{\\frac{1}{2}}``, ``\\boxed{\\{1, 2, 3\\}}``) are
    removed in full instead of leaving a dangling ``{2}}``-style fragment behind — math
    reasoning traces box fractions, sets, and intervals often enough that this is not an
    edge case. A ``\\boxed{`` with no matching ``}`` (a truncated trace) is left as-is
    rather than silently consuming the rest of the string.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        m = _BOXED_START_RE.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        depth = 1
        j = m.end()
        while j < n and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            i = j  # balanced: drop the whole \boxed{...} span
        else:
            out.append(text[i : m.end()])  # unbalanced: keep the literal "\boxed{"
            i = m.end()
    return "".join(out)


def build_record(
    prompt: Any,
    cot: Any,
    answer: Any,
    *,
    prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
    min_cot_len: int = 5,
) -> dict | None:
    """Build a single chat-format SFT record from a ``(prompt, cot, answer)`` triple.

    Returns a ``{"messages": [user, assistant]}`` dict, or ``None`` when the trace is
    empty / ``"nan"`` / shorter than *min_cot_len* characters (these rows are dropped).

    The assistant turn is ``<stripped cot>\\n</think>\\n\\boxed{answer}``; the opening
    ``<think>`` is expected to be inserted by the chat template (see
    :func:`render_chat`), keeping this function tokenizer-agnostic.
    """
    cot = str(cot)
    if not cot or cot == "nan" or len(cot.strip()) < min_cot_len:
        return None
    cot_cleaned = strip_boxed(cot).rstrip()
    user_content = str(prompt) + prompt_suffix
    asst_content = cot_cleaned + f"\n</think>\n\\boxed{{{str(answer)}}}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": asst_content},
        ]
    }


def build_records(
    rows: Iterable[Mapping[str, Any]] | "pandas.DataFrame",  # noqa: F821
    *,
    prompt_key: str = "prompt",
    cot_key: str = "generated_cot",
    answer_key: str = "answer",
    type_key: str = "type",
    prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
    min_cot_len: int = 5,
) -> tuple[list[dict], list[str]]:
    """Vectorised :func:`build_record` over an iterable of mappings or a DataFrame.

    Returns ``(records, types)`` — two parallel lists, where ``types[i]`` is the
    problem-type label of ``records[i]`` (consumed by the stratified sampler, see
    :mod:`tracedistill.sampling`). Rows with an unusable trace are skipped, so the
    returned lists may be shorter than the input.

    Accepts either a list of dict-like rows or a pandas ``DataFrame`` (detected via
    ``.iterrows``), so callers don't need pandas to use it.
    """
    iterator: Iterable[Mapping[str, Any]]
    if hasattr(rows, "iterrows"):
        iterator = (row for _, row in rows.iterrows())  # type: ignore[union-attr]
    else:
        iterator = rows  # type: ignore[assignment]

    records: list[dict] = []
    types: list[str] = []
    for row in iterator:
        rec = build_record(
            row[prompt_key],
            row[cot_key],
            row[answer_key],
            prompt_suffix=prompt_suffix,
            min_cot_len=min_cot_len,
        )
        if rec is None:
            continue
        records.append(rec)
        types.append(str(row[type_key]))
    return records, types
