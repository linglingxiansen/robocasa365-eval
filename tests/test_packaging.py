import json
from pathlib import Path

import pytest

from scripts.audit_repository import audit
from scripts.export_artifacts import sanitize_json
from scripts.verify_reference import verify

ROOT = Path(__file__).resolve().parents[1]


def test_reference_counts():
    assert verify(ROOT / "results/reference_200k")["successes"] == 1361


def test_reference_rejects_modified_stats(tmp_path):
    import shutil
    root = tmp_path / "reference"
    shutil.copytree(ROOT / "results/reference_200k", root)
    path = next((root / "evals/pretrain").glob("*/stats.json"))
    data = json.loads(path.read_text())
    data["num_episodes"] = 49
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify(root)


def test_export_removes_cached_locations_without_changing_model_shape():
    cfg = {"_name_or_path": "private-location", "text_config": {"hidden_size": 2560, "name_or_path": "private-location"},
           "added_tokens_decoder": {"100": {"content": "<action>"}}}
    clean = sanitize_json(cfg)
    assert clean["text_config"] == {"hidden_size": 2560}
    assert clean["added_tokens_decoder"] == cfg["added_tokens_decoder"]
    assert "private-location" not in json.dumps(clean)


def test_audit_detects_private_ip_and_does_not_echo_it(tmp_path):
    private_ip = ".".join(["10", "23", "45", "67"])
    (tmp_path / "config.txt").write_text("host=" + private_ip)
    report = audit(tmp_path)
    assert not report["passed"]
    assert report["findings"][0]["rule"] == "private-ipv4"
    assert private_ip not in json.dumps(report)


def test_audit_rejects_symlink(tmp_path):
    (tmp_path / "external").symlink_to(tmp_path.parent)
    assert not audit(tmp_path)["passed"]
