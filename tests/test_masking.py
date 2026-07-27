"""Tests for completion-only chat rendering and token-boundary masking."""

import pytest

from tracedistill.masking import render_prompt_completion, tokenize_with_masked_prompt


class ChatTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=True,
    ):
        assert tokenize is False
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return text


class CharacterTokenizer:
    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        truncation=False,
        max_length=None,
        return_offsets_mapping=False,
    ):
        assert add_special_tokens is False
        limit = max_length if truncation else len(text)
        text = text[:limit]
        result = {
            "input_ids": [ord(char) for char in text],
            "attention_mask": [1] * len(text),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class BoundaryMergingTokenizer:
    """Slow-tokenizer stub whose final prompt token merges with the completion."""

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        truncation=False,
        max_length=None,
        return_offsets_mapping=False,
    ):
        if return_offsets_mapping:
            raise NotImplementedError
        ids = [1, 2] if text == "ab" else [1, 9, 3]
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_render_supports_system_and_user_prompt():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    prompt, completion = render_prompt_completion(ChatTokenizer(), messages)
    assert prompt == "<system>rules<user>question<assistant>"
    assert completion == "answer"


def test_render_rejects_unsafe_conversation_shape():
    with pytest.raises(ValueError, match="assistant"):
        render_prompt_completion(
            ChatTokenizer(),
            [{"role": "user", "content": "q"}, {"role": "user", "content": "not a target"}],
        )


def test_offset_masking_supervises_only_completion():
    row = tokenize_with_masked_prompt(
        CharacterTokenizer(),
        "prompt:",
        "answer",
        max_length=64,
    )
    assert row is not None
    assert row["labels"][:7] == [-100] * 7
    assert row["labels"][7:] == [ord(char) for char in "answer"]
    assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["labels"])


def test_boundary_merging_token_is_masked_for_slow_tokenizer():
    row = tokenize_with_masked_prompt(
        BoundaryMergingTokenizer(),
        "ab",
        "c",
        max_length=8,
    )
    assert row is not None
    assert row["input_ids"] == [1, 9, 3]
    assert row["labels"] == [-100, -100, 3]


def test_truncation_that_removes_completion_returns_none():
    assert (
        tokenize_with_masked_prompt(
            CharacterTokenizer(),
            "long prompt",
            "answer",
            max_length=5,
        )
        is None
    )


def test_invalid_length_and_empty_completion_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        tokenize_with_masked_prompt(CharacterTokenizer(), "p", "c", max_length=0)
    with pytest.raises(ValueError, match="empty"):
        tokenize_with_masked_prompt(CharacterTokenizer(), "p", "", max_length=8)
