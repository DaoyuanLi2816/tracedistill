"""Tests for the config-driven CLI surface (no torch needed)."""

import os

import pytest
import yaml

from tracedistill.run import RunConfig, load_config

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


@pytest.mark.parametrize("name", ["quickstart.yaml", "reproduce_competition.yaml"])
def test_shipped_example_configs_load(name):
    # The example YAMLs must stay valid (no unknown/typo'd keys).
    cfg = load_config(os.path.join(EXAMPLES, name))
    assert cfg.base_model and cfg.data_path and cfg.hard_types
