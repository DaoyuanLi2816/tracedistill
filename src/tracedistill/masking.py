"""Completion-only rendering and token masking.

The training objective should supervise the assistant's reasoning trace and final
answer, never the prompt. Chat templates make that boundary easy to get subtly wrong:
rendering the prompt and the full conversation separately can be text-composable while
their tokenizations differ at the join. This module therefore finds the boundary in the
tokenization of the *combined* text and masks every token that starts before the
assistant completion.

The functions are dependency-light and work with any tokenizer exposing the standard
Hugging Face call interface.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["render_prompt_completion", "tokenize_with_masked_prompt"]


def _apply_chat_template(tokenizer, messages: Sequence[dict], **kwargs) -> str:
    """Render a chat template while supporting tokenizers without ``enable_thinking``."""
    try:
        return tokenizer.apply_chat_template(list(messages), **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(list(messages), **kwargs)


def render_prompt_completion(
    tokenizer,
    messages: Sequence[dict],
    *,
    enable_thinking: bool = True,
) -> tuple[str, str]:
    """Render a conversation as a prompt plus an assistant completion.

    ``messages`` may contain system and user turns, but its final message must be the
    assistant target. The prompt is rendered with ``add_generation_prompt=True`` and
    the full conversation without it. The function fails closed when the chat template
    does not compose as a strict prefix, because an uncertain boundary must not silently
    leak prompt tokens into the loss.
    """
    if len(messages) < 2:
        raise ValueError("Expected at least one prompt message and one assistant message.")
    if messages[-1].get("role") != "assistant":
        raise ValueError("The final message must have role='assistant'.")

    common = {"tokenize": False, "enable_thinking": enable_thinking}
    prompt = _apply_chat_template(
        tokenizer,
        messages[:-1],
        add_generation_prompt=True,
        **common,
    )
    full = _apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=False,
        **common,
    )
    if not isinstance(prompt, str) or not isinstance(full, str):
        raise TypeError("The chat template must return text when tokenize=False.")
    if not full.startswith(prompt):
        raise ValueError(
            "Rendered prompt is not a prefix of the full conversation; completion-only "
            "masking cannot establish a safe assistant boundary for this chat template."
        )
    completion = full[len(prompt) :]
    if not completion:
        raise ValueError("The rendered assistant completion is empty.")
    return prompt, completion


def _tokenize_with_offsets(
    tokenizer,
    text: str,
    *,
    max_length: int,
) -> tuple[list[int], list[int], list[tuple[int, int]]] | None:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
    except (TypeError, NotImplementedError, ValueError):
        return None

    offsets = encoded.get("offset_mapping")
    if offsets is None:
        return None
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
    return input_ids, attention_mask, [tuple(pair) for pair in offsets]


def _tokenize_without_offsets(
    tokenizer,
    prompt: str,
    full: str,
    *,
    max_length: int,
) -> tuple[list[int], list[int], int]:
    """Fallback for slow tokenizers.

    If a token straddles the text boundary, it is included in the masked prefix. This
    can discard one completion token, but it guarantees that no token containing prompt
    text contributes to the objective.
    """
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    encoded = tokenizer(
        full,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))

    common = 0
    for prompt_id, full_id in zip(prompt_ids, input_ids, strict=False):
        if prompt_id != full_id:
            break
        common += 1
    # A mismatch before the end of the prompt indicates a token that changed at the
    # prompt/completion join. Mask that boundary token as well.
    completion_start = common + int(common < len(prompt_ids) and common < len(input_ids))
    return input_ids, attention_mask, completion_start


def tokenize_with_masked_prompt(
    tokenizer,
    prompt: str,
    completion: str,
    *,
    max_length: int,
) -> dict[str, list[int]] | None:
    """Tokenize ``prompt + completion`` with labels only on completion tokens.

    Tokens wholly inside the prompt, or straddling the prompt/completion boundary, are
    assigned ``-100``. Returns ``None`` when truncation leaves no supervised completion
    token. The input text is tokenized only once when offset mappings are available,
    avoiding the boundary drift caused by independently tokenizing the two strings.
    """
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if not completion:
        raise ValueError("completion must not be empty.")

    full = prompt + completion
    with_offsets = _tokenize_with_offsets(tokenizer, full, max_length=max_length)
    if with_offsets is not None:
        input_ids, attention_mask, offsets = with_offsets
        boundary = len(prompt)
        completion_start = len(input_ids)
        for index, (start, end) in enumerate(offsets):
            # (0, 0) is commonly used for special tokens. Although special tokens are
            # disabled here, mask any such token defensively.
            if end > start and start >= boundary:
                completion_start = index
                break
    else:
        input_ids, attention_mask, completion_start = _tokenize_without_offsets(
            tokenizer,
            prompt,
            full,
            max_length=max_length,
        )

    if completion_start >= len(input_ids):
        return None
    labels = list(input_ids)
    labels[:completion_start] = [-100] * completion_start
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
