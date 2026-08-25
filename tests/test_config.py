from __future__ import annotations

import json

import pytest

from fpl_brain.config import ConfigError, load_config


def test_config_json_root_must_be_object(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ConfigError, match="root"):
        load_config(path)
