"""Focused tests for the optional PyTorch/TRL training layer."""

import pytest

torch = pytest.importorskip("torch")
training = pytest.importorskip("tracedistill.training")
PromptMaskedCollator = training.PromptMaskedCollator


def test_prompt_masked_collator_preserves_labels_and_masks_padding():
    collator = PromptMaskedCollator(pad_token_id=0)
    batch = collator(
        [
            {
                "input_ids": [10, 11, 12],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 11, 12],
            },
            {
                "input_ids": [20, 21],
                "attention_mask": [1, 1],
                "labels": [-100, 21],
            },
        ]
    )
    assert batch["input_ids"].tolist() == [[10, 11, 12], [20, 21, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[-100, 11, 12], [-100, 21, -100]]


def test_prompt_masked_collator_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty"):
        PromptMaskedCollator(pad_token_id=0)([])
