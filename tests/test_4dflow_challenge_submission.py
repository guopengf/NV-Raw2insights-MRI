import argparse
import contextlib
import hashlib
import io
import inspect
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from scripts.tools import run_4dflow_challenge_submission as submission


class ChallengeSubmissionTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    FAMILIES = ("joint_epoch100", "joint_channel_epoch260")
    STAGES = ("full", "package", "preflight", "recover", "smoke")

    @classmethod
    def launcher_text(cls, family: str, stage: str) -> str:
        return (cls.ROOT / f"run_raw2ins_4dflow_{family}_challenge_{stage}.slurm").read_text()

    def test_both_launcher_families_use_canonical_checkout(self):
        expected = {
            f"run_raw2ins_4dflow_{family}_challenge_{stage}.slurm"
            for family in self.FAMILIES
            for stage in self.STAGES
        }
        actual = {
            path.name
            for path in self.ROOT.glob("run_raw2ins_4dflow_joint*_challenge_*.slurm")
        }
        self.assertTrue(expected.issubset(actual), expected - actual)
        for family in self.FAMILIES:
            for stage in self.STAGES:
                with self.subTest(family=family, stage=stage):
                    text = self.launcher_text(family, stage)
                    self.assertIn("cd /workspace/code/NV-Raw2insights-MRI-fork", text)
                    self.assertNotIn("export GIT_DIR=", text)
                    self.assertNotIn("export GIT_WORK_TREE=", text)

    def test_smoke_launchers_match_one_case_per_shard(self):
        for family in self.FAMILIES:
            with self.subTest(family=family):
                text = self.launcher_text(family, "smoke")
                self.assertIn("#SBATCH --array=0-5%6", text)
                self.assertIn("#SBATCH --partition=batch,batch_short,interactive", text)
                self.assertNotIn("#SBATCH --exclusive", text)
                self.assertNotIn("#SBATCH --cpus-per-task", text)
                self.assertIn("#SBATCH --gpus-per-node=1", text)
                self.assertIn("    --nproc 1 \\", text)
                self.assertIn("    --batch-size 4 \\", text)
                self.assertIn("    --num-workers 0", text)

    def test_full_launchers_keep_validated_provenance_and_runtime_settings(self):
        expected = {
            "joint_epoch100": {
                "workers": 4,
                "config": "configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_batch_windowed_h5_epoch100_inference.json",
                "checkpoint": "nv_raw2insights_mri_small_ft_4dflow_joint_batch_3d_flowvn_multiplane_windowed_h5_epoch100.pt",
                "checkpoint_sha256": "65af7ea1d0004afa4f15fac0ceb4c1ab6a53984c40fa1fe4581d059db8a6365d",
                "config_sha256": "dbb324f41b3c9fafa5728a9f40d727a68cb6c596f8264719fea70ecadeb9cbb2",
                "epoch": 100,
                "global_step": 17606,
                "wandb_run_id": "kqi05x9h",
            },
            "joint_channel_epoch260": {
                "workers": 0,
                "config": "configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_channel_pg.json",
                "checkpoint": "nv_raw2insights_mri_small_ft_4dflow_joint_channel_3d_flowvn_multiplane_epoch260.pt",
                "checkpoint_sha256": "154ffdd3ea68512d699ce7d65448122fdc6fa7ecff8f050bcd7a6301359f3817",
                "config_sha256": "5d784d2d07f10da41c7b0465f034449c061a61b26446789cdf966d1ebab9280e",
                "epoch": 260,
                "global_step": 45941,
                "wandb_run_id": "v9yybr91",
            },
        }
        for family, values in expected.items():
            with self.subTest(family=family):
                text = self.launcher_text(family, "full")
                self.assertIn("#SBATCH --array=0-5%6", text)
                self.assertIn("#SBATCH --partition=interactive,batch", text)
                self.assertIn("#SBATCH --gpus-per-node=8", text)
                self.assertNotIn("#SBATCH --exclusive", text)
                self.assertNotIn("#SBATCH --cpus-per-task", text)
                self.assertIn("    --nproc 8 \\", text)
                self.assertIn("    --batch-size 4 \\", text)
                self.assertIn(f"    --num-workers {values['workers']}", text)
                self.assertIn(f"--config {values['config']}", text)
                self.assertIn(values["checkpoint"], text)
                self.assertIn(
                    f"--expected-checkpoint-sha256 {values['checkpoint_sha256']}", text
                )
                self.assertIn(
                    f"--expected-config-sha256 {values['config_sha256']}", text
                )
                self.assertIn(f"--expected-epoch {values['epoch']}", text)
                self.assertIn(f"--expected-global-step {values['global_step']}", text)
                self.assertIn(f"--expected-wandb-run-id {values['wandb_run_id']}", text)

    def test_recovery_launchers_keep_single_process_settings(self):
        for family in self.FAMILIES:
            with self.subTest(family=family):
                text = self.launcher_text(family, "recover")
                self.assertNotIn("#SBATCH --exclusive", text)
                self.assertNotIn("#SBATCH --cpus-per-task", text)
                self.assertIn("    --nproc 1 \\", text)
                self.assertIn("    --batch-size 4 \\", text)
                self.assertIn("    --num-workers 0", text)

        channel_text = self.launcher_text("joint_channel_epoch260", "recover")
        self.assertIn("    --data-base /data/CMRx4DFlow2026-ChallengeData \\", channel_text)
        self.assertIn("    --skip-preflight", channel_text)

    def test_epoch100_launchers_use_frozen_config_snapshot(self):
        config_name = (
            "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_"
            "joint_batch_windowed_h5_epoch100_inference.json"
        )
        frozen = self.ROOT / "configs" / config_name
        current = (
            self.ROOT
            / "configs"
            / "nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_"
            "joint_batch_windowed_h5_pg.json"
        )
        self.assertEqual(
            hashlib.sha256(frozen.read_bytes()).hexdigest(),
            "dbb324f41b3c9fafa5728a9f40d727a68cb6c596f8264719fea70ecadeb9cbb2",
        )
        self.assertEqual(json.loads(frozen.read_text())["num_epochs"], 100)
        self.assertEqual(json.loads(current.read_text())["num_epochs"], 300)
        for stage in ("preflight", "smoke", "full"):
            with self.subTest(stage=stage):
                text = self.launcher_text("joint_epoch100", stage)
                self.assertIn(f"--config configs/{config_name}", text)

    def test_provenance_expectations_have_no_mode_specific_defaults(self):
        signature = inspect.signature(submission.verify_provenance)
        expected_names = (
            "expected_checkpoint_sha256",
            "expected_config_sha256",
            "expected_inference_sha256",
            "expected_exporter_sha256",
            "expected_epoch",
            "expected_global_step",
            "expected_wandb_run_id",
        )
        for name in expected_names:
            with self.subTest(name=name):
                parameter = signature.parameters[name]
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_shard_inventory_matches_validation_contract(self):
        self.assertEqual(sum(spec["full_cases"] for spec in submission.SHARDS.values()), 112)
        task_counts = {}
        for spec in submission.SHARDS.values():
            task_counts[spec["task"]] = task_counts.get(spec["task"], 0) + spec["full_cases"]
        self.assertEqual(task_counts, {"TaskR1R2": 32, "TaskS1": 40, "TaskS2": 40})
        self.assertEqual(
            {spec["smoke_acceleration"] for spec in submission.SHARDS.values()},
            {10, 20, 30, 40, 50},
        )

    def test_joint_manifest_has_one_input_and_four_reconstructions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = {
                "anatomy": "Aorta",
                "center": "Center001",
                "scanner": "Scanner",
                "patient": "P001",
                "acceleration": 20,
                "kspace": "/data/kdata_ktGaussian20.mat",
                "mask": "/data/usmask_ktGaussian20.mat",
                "coilmap": "/data/coilmap.mat",
                "segmask": "/data/segmask.mat",
            }
            inputs, reconstructions = submission.build_inference_manifest(
                [record], root / "jsons", "TaskR1R2", [0, 1, 2, 3]
            )
            self.assertEqual(len(inputs), 1)
            self.assertEqual(len(reconstructions), 4)
            payload = json.loads(Path(inputs[0]["json"]).read_text())
            self.assertTrue(payload["joint_encodings"])
            self.assertEqual(payload["encoding_indices"], [0, 1, 2, 3])
            self.assertNotIn("encoding_idx", payload)
            self.assertEqual(
                [item["stem"].rsplit("__", 1)[-1] for item in reconstructions],
                ["enc0", "enc1", "enc2", "enc3"],
            )

    def test_joint_config_and_effective_worker_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            effective = root / "effective.json"
            for mode in ("batch", "channel"):
                with self.subTest(mode=mode):
                    source.write_text(
                        json.dumps(
                            {
                                "num_workers": 8,
                                "phase3": {
                                    "joint_encoding": {
                                        "enabled": True,
                                        "mode": mode,
                                        "count": 4,
                                        "order": [0, 1, 2, 3],
                                    }
                                },
                            }
                        )
                    )
                    self.assertEqual(
                        submission.joint_encoding_order(source), [0, 1, 2, 3]
                    )
                    submission.write_effective_config(source, effective, 0, 4)
                    payload = json.loads(effective.read_text())
                    self.assertEqual(payload["num_workers"], 0)
                    self.assertEqual(payload["batch_size"], 4)

    def test_package_shards_validates_and_writes_expected_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            submission_root = run_root / "submission"
            provenance = {
                "paths": {"checkpoint": "/checkpoint.pt"},
                "sha256": {"checkpoint": "checkpoint-sha"},
                "checkpoint_metadata": {"epoch": 260, "global_step": 45941},
                "git_head": "head",
                "git_status": "",
            }

            for shard_index, spec in submission.SHARDS.items():
                relpaths = []
                for case_index in range(spec["full_cases"]):
                    relative = (
                        Path(spec["task"])
                        / "ValidationSet"
                        / spec["anatomies"][0]
                        / "Center"
                        / "Scanner"
                        / f"Patient{case_index:03d}"
                        / "img_ktGaussian10.npz"
                    )
                    path = submission_root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        path,
                        coords=np.array([[0, 0, 0, 0, 0]], dtype=np.int32),
                        data=np.array([1 + 2j], dtype=np.complex64),
                        shape=np.array([4, 1, 1, 1, 1], dtype=np.int64),
                    )
                    relpaths.append(str(relative))

                summary_root = run_root / "work" / spec["work_name"]
                summary_root.mkdir(parents=True)
                (summary_root / "shard_summary.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "mode": "full",
                            "task": spec["task"],
                            "anatomies": spec["anatomies"],
                            "case_count": spec["full_cases"],
                            "submission_count": spec["full_cases"],
                            "expected_submission_relpaths": relpaths,
                            "provenance": provenance,
                            "shard_index": shard_index,
                        }
                    )
                )

            with contextlib.redirect_stdout(io.StringIO()):
                submission.package_shards(
                    argparse.Namespace(
                        run_root=run_root,
                        artifact_tag="joint_integration_test",
                        skip_preflight=True,
                        skip_combined_zip=True,
                    )
                )

            artifact_manifest = json.loads(
                (run_root / "artifacts" / "artifact_manifest.json").read_text()
            )
            self.assertEqual(artifact_manifest["submission_count"], 112)
            self.assertEqual(
                artifact_manifest["task_counts"],
                {"TaskR1R2": 32, "TaskS1": 40, "TaskS2": 40},
            )
            self.assertEqual(len(artifact_manifest["zip_artifacts"]), 3)
            self.assertFalse(artifact_manifest["combined_zip_included"])
            self.assertFalse((run_root / "artifacts" / "Submission.zip").exists())

    def test_package_shards_rejects_unsafe_artifact_tag(self):
        with self.assertRaises(ValueError):
            submission.package_shards(
                argparse.Namespace(run_root=Path("/unused"), artifact_tag="epoch300/bad")
            )


if __name__ == "__main__":
    unittest.main()
