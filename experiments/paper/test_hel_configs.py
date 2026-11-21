"""hel_configs.py writes one config per experiment, under its own name, with the agreed HEL settings."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent / "scripts" / "hel_configs.py"


def config(name, method, dims=3):
    return {"batch_size": 4, "phase3": {"backbone": {"spatial_dims": dims}},
            "paper": {"method": method, "name": name, "protected_roots": [], "loader": {}, "online_undersampling": {"enabled": True}}}


def test_generated_configs(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "h5" / "cases").mkdir(parents=True)
    for name, method, dims in (("ablation_A0", "ours", 2), ("ablation_A1", "ours", 3), ("ours_full", "ours", 3), ("flowvn", "flowvn", 3), ("cg_sense", "cg_sense", 3)):
        (tmp_path / "configs" / f"{name}.json").write_text(json.dumps(config(name, method, dims)))
    for weights in ("w3d.pt", "w2d.pt"):
        (tmp_path / weights).write_bytes(b"")
    subprocess.run([sys.executable, str(SCRIPT), "--run-root", str(tmp_path), "--h5-root", str(tmp_path / "h5"), "--accumulate", "ablation_A1",
                    "--bucket-steps", "8", "--pretrained", str(tmp_path / "w3d.pt"), "--pretrained-2d", str(tmp_path / "w2d.pt")], check=True)
    out = {p.name: json.loads(p.read_text()) for p in (tmp_path / "configs_bench").iterdir()}
    assert sorted(out) == ["ablation_A0.json", "ablation_A1.json", "cg_sense.json", "flowvn.json", "ours_full.json"]
    paper = {name[:-5]: cfg["paper"] for name, cfg in out.items()}
    assert all(p["name"] == name and p["train_backend"] == "windowed_hdf5" for name, p in paper.items())
    assert [Path(paper[n]["pretrained"]).name for n in ("ablation_A0", "ablation_A1", "ours_full")] == ["w2d.pt", "w3d.pt", "w3d.pt"]
    assert "pretrained" not in paper["flowvn"] and "loader" in paper["cg_sense"] and "bucket_steps" not in paper["cg_sense"]["loader"]
    assert (paper["ours_full"]["loader"]["stack"], paper["ours_full"]["gradient_accumulation"]) == (2, 1)
    assert (paper["ablation_A1"]["loader"]["stack"], paper["ablation_A1"]["gradient_accumulation"]) == (1, 2)
    assert all(paper[n]["loader"]["bucket_steps"] == 8 for n in ("ablation_A0", "ours_full", "flowvn"))
    assert out["ours_full.json"]["batch_size"] == 8 and paper["ours_full"]["checkpoint_permanent_every"] == 100
    assert paper["flowvn"]["checkpoint_permanent_every"] == 10 and not paper["flowvn"]["online_undersampling"]["enabled"]
