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
    def test_shard_inventory_matches_validation_contract(self):
        self.assertEqual(sum(spec["full_cases"] for spec in submission.SHARDS.values()), 112)
        task_counts = {}
        for spec in submission.SHARDS.values():
            task_counts[spec["task"]] = task_counts.get(spec["task"], 0) + spec["full_cases"]
        self.assertEqual(task_counts, {"TaskR1R2": 32, "TaskS1": 40, "TaskS2": 40})

    def test_package_shards_validates_and_writes_expected_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            submission_root = run_root / "submission"
            provenance = {
                "paths": {"checkpoint": "/checkpoint.pt"},
                "sha256": {"checkpoint": "checkpoint-sha"},
                "checkpoint_metadata": {"epoch": 300, "global_step": 51937},
                "git_head": "head",
                "git_status": "",
            }
            (run_root / "preflight.json").write_text(
                json.dumps({"status": "complete", "case_count": 112, "provenance": provenance})
            )

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
                    argparse.Namespace(run_root=run_root, artifact_tag="epoch300")
                )

            artifact_manifest = json.loads(
                (run_root / "artifacts" / "artifact_manifest.json").read_text()
            )
            self.assertEqual(artifact_manifest["submission_count"], 112)
            self.assertEqual(
                artifact_manifest["task_counts"],
                {"TaskR1R2": 32, "TaskS1": 40, "TaskS2": 40},
            )
            self.assertEqual(len(artifact_manifest["zip_artifacts"]), 4)
            with zipfile.ZipFile(run_root / "artifacts" / "Submission.zip") as archive:
                self.assertEqual(len(archive.namelist()), 112)
                self.assertIsNone(archive.testzip())

    def test_package_shards_rejects_unsafe_artifact_tag(self):
        with self.assertRaises(ValueError):
            submission.package_shards(
                argparse.Namespace(run_root=Path("/unused"), artifact_tag="epoch300/bad")
            )


if __name__ == "__main__":
    unittest.main()
