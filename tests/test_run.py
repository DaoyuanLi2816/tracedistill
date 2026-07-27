"""Tests for the config-driven CLI surface (no torch needed)."""

import os

import pandas as pd
import pytest
import yaml

from tracedistill.run import RunConfig, load_config, main, summarize_data

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, "examples", "configs")


def _write(tmp_path, mapping):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return str(p)


def test_minimal_config_loads(tmp_path):
    cfg = load_config(
        _write(tmp_path, {"base_model": "m", "data_path": "d.csv", "hard_types": ["t"]})
    )
    assert isinstance(cfg, RunConfig)
    assert cfg.base_model == "m" and cfg.lora_rank == 32  # default applied


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(
            _write(
                tmp_path,
                {"base_model": "m", "data_path": "d", "hard_types": ["t"], "lr": 1e-4},
            )
        )


def test_missing_required_keys_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, {"base_model": "m"}))


def test_non_mapping_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_config(str(p))


def test_phase_overrides_pass_through(tmp_path):
    cfg = load_config(
        _write(
            tmp_path,
            {
                "base_model": "m",
                "data_path": "d",
                "hard_types": ["t"],
                "phase1": {"learning_rate": 1e-3},
                "phase2": {"num_train_epochs": 2},
            },
        )
    )
    assert cfg.phase1 == {"learning_rate": 1e-3}
    assert cfg.phase2 == {"num_train_epochs": 2}


def test_unknown_phase_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown phase1 keys"):
        load_config(
            _write(
                tmp_path,
                {
                    "base_model": "m",
                    "data_path": "d",
                    "hard_types": ["t"],
                    "phase1": {"learnng_rate": 1e-4},
                },
            )
        )


def test_dry_run_summary_needs_no_training_stack(tmp_path, capsys):
    data_path = tmp_path / "data.csv"
    pd.DataFrame(
        [
            {"prompt": "hard", "generated_cot": "reasoning", "answer": 1, "type": "hard"},
            {"prompt": "easy1", "generated_cot": "reasoning", "answer": 2, "type": "easy"},
            {"prompt": "easy2", "generated_cot": "reasoning", "answer": 3, "type": "easy"},
        ]
    ).to_csv(data_path, index=False)
    cfg_path = _write(
        tmp_path,
        {"base_model": "model", "data_path": str(data_path), "hard_types": ["hard"]},
    )

    summary = summarize_data(load_config(cfg_path))
    assert summary["rows"] == 3
    assert summary["usable_records"] == 3
    assert summary["completion_only_loss"] is True

    main(["--cfg", cfg_path, "--dry-run"])
    output = capsys.readouterr().out
    assert '"completion_only_loss": true' in output


@pytest.mark.parametrize("name", ["quickstart.yaml", "reproduce_competition.yaml"])
def test_shipped_example_configs_load(name):
    # The example YAMLs must stay valid (no unknown/typo'd keys).
    cfg = load_config(os.path.join(EXAMPLES, name))
    assert cfg.base_model and cfg.data_path and cfg.hard_types
