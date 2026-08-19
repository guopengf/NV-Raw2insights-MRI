import argparse
import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from scripts.tools import run_4dflow_challenge_submission as submission


class ChallengeSubmissionTest(unittest.TestCase):
    def test_smoke_launcher_matches_one_case_per_shard(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "run_raw2ins_4dflow_joint_channel_epoch260_challenge_smoke.slurm"
        )
        text = launcher.read_text()
        self.assertIn("#SBATCH --array=0-5%6", text)
        self.assertIn("#SBATCH --partition=batch,batch_short,interactive", text)
        self.assertNotIn("#SBATCH --exclusive", text)
        self.assertNotIn("#SBATCH --cpus-per-task", text)
        self.assertIn("#SBATCH --gpus-per-node=1", text)
        self.assertIn("    --nproc 1 \\", text)
        self.assertIn("    --batch-size 4 \\", text)
        self.assertIn("    --num-workers 0", text)

    def test_full_launcher_uses_approved_scheduler_and_loader_settings(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "run_raw2ins_4dflow_joint_channel_epoch260_challenge_full.slurm").read_text()
        self.assertIn("#SBATCH --array=0-5%6", text)
        self.assertIn("#SBATCH --partition=interactive,batch", text)
        self.assertIn("#SBATCH --gpus-per-node=8", text)
        self.assertNotIn("#SBATCH --exclusive", text)
        self.assertNotIn("#SBATCH --cpus-per-task", text)
        self.assertIn("    --nproc 8 \\", text)
        self.assertIn("    --batch-size 4 \\", text)
        self.assertIn("    --num-workers 0", text)
        self.assertIn("--config configs/nv_raw2insights_mri_small_4dflow_3d_flowvn_multiplane_joint_channel_pg.json", text)
        self.assertIn("nv_raw2insights_mri_small_ft_4dflow_joint_channel_3d_flowvn_multiplane_epoch260.pt", text)
        self.assertIn("--expected-checkpoint-sha256 154ffdd3ea68512d699ce7d65448122fdc6fa7ecff8f050bcd7a6301359f3817", text)
        self.assertIn("--expected-config-sha256 5d784d2d07f10da41c7b0465f034449c061a61b26446789cdf966d1ebab9280e", text)
        self.assertIn("--expected-epoch 260", text)
        self.assertIn("--expected-global-step 45941", text)
        self.assertIn("--expected-wandb-run-id v9yybr91", text)

    def test_recovery_launcher_keeps_single_process_settings(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "run_raw2ins_4dflow_joint_channel_epoch260_challenge_recover.slurm").read_text()
        self.assertNotIn("#SBATCH --exclusive", text)
        self.assertNotIn("#SBATCH --cpus-per-task", text)
        self.assertIn("    --nproc 1 \\", text)
        self.assertIn("    --batch-size 4 \\", text)
        self.assertIn("    --num-workers 0", text)

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
            source.write_text(
                json.dumps(
                    {
                        "num_workers": 8,
                        "phase3": {
                            "joint_encoding": {
                                "enabled": True,
                                "mode": "channel",
                                "count": 4,
                                "order": [0, 1, 2, 3],
                            }
                        },
                    }
                )
            )
            self.assertEqual(submission.joint_encoding_order(source), [0, 1, 2, 3])
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
                        artifact_tag="small_joint_channel_epoch260",
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
