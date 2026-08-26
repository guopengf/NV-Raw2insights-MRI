from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from utils import load_config, save_args_to_file_json


def test_nested_config_defaults_are_not_serialized(tmp_path):
    augmentation = {
        "enabled": True,
        "flip": {"prob": 0.25, "axes": ["z", "y"]},
        "shift": {"prob": 0.25, "max_pixels": [3, 8]},
        "contrast": {"prob": 0.2, "gamma": [0.8, 1.2]},
    }
    source = {
        "data_aug": True,
        "four_dflow_augmentation": augmentation,
        "four_dflow_storage": {
            "backend": "raw_mat",
            "train_backend": "windowed_hdf5",
            "val_backend": "raw_mat",
        },
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    config = load_config(source_path)

    assert config.data_path_train is None
    assert not hasattr(config.four_dflow_augmentation, "data_path_train")
    assert not hasattr(config.four_dflow_augmentation.flip, "pp_norm")

    saved_path = tmp_path / "saved.json"
    save_args_to_file_json(config, saved_path)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved["four_dflow_augmentation"] == augmentation
    assert saved["four_dflow_storage"] == source["four_dflow_storage"]
    assert saved["data_path_train"] is None
