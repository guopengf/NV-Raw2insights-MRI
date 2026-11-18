import importlib.util
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/tools/profile_unified_validation_memory.py"
SPEC = importlib.util.spec_from_file_location("memory_profile", SOURCE)
MEM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEM)


def sample():
    return dict(cgroup={"memory.max": 180*MEM.GIB, "memory.current": 179*MEM.GIB,
        "memory.peak": 180*MEM.GIB, "stat": {"anon": 20*MEM.GIB, "file": 155*MEM.GIB,
        "kernel": 4*MEM.GIB, "shmem": 0, "file_dirty": 0, "file_writeback": 0},
        "events": {"max": 0, "oom": 0, "oom_kill": 0}},
        processes=[{"pid": 10, "VmRSS": 22*MEM.GIB, "VmHWM": 23*MEM.GIB,
                    "RssAnon": 20*MEM.GIB}],
        gpu=[{"used_bytes": 7*MEM.GIB, "total_bytes": 80*MEM.GIB}])


class MemoryTests(unittest.TestCase):
    def test_file_cache_not_confused_with_anonymous_memory(self):
        result = MEM.summarize([sample() for _ in range(10)])
        self.assertTrue(result["memory_gate_passed"])
        self.assertEqual(result["peak_anon_bytes"], 20*MEM.GIB)
        self.assertEqual(result["peak_file_cache_bytes"], 155*MEM.GIB)
        self.assertEqual(result["peak_non_cache_required_bytes"], 24*MEM.GIB)

    def test_non_cache_headroom_refuses(self):
        rows = [sample() for _ in range(10)]
        rows[-1]["cgroup"]["stat"]["anon"] = 150*MEM.GIB
        self.assertFalse(MEM.summarize(rows)["memory_gate_passed"])

    def test_dirty_and_shared_pages_count_conservatively(self):
        row = sample()
        row["cgroup"]["stat"].update(shmem=2*MEM.GIB, file_dirty=3*MEM.GIB,
                                      file_writeback=4*MEM.GIB)
        self.assertEqual(MEM.required_bytes(row), 33*MEM.GIB)

    def test_oom_event_refuses(self):
        rows = [sample() for _ in range(10)]
        rows[-1]["cgroup"]["events"]["oom_kill"] = 1
        self.assertFalse(MEM.summarize(rows)["memory_gate_passed"])

    def test_cache_reclamation_limit_event_is_reported_not_oom(self):
        rows = [sample() for _ in range(10)]
        rows[-1]["cgroup"]["events"]["max"] = 12
        result = MEM.summarize(rows)
        self.assertTrue(result["memory_gate_passed"])
        self.assertEqual(result["events_delta"]["max"], 12)

    def test_gpu_headroom_refuses(self):
        rows = [sample() for _ in range(10)]
        rows[-1]["gpu"][0]["used_bytes"] = 70*MEM.GIB
        self.assertFalse(MEM.summarize(rows)["memory_gate_passed"])

    def test_wrong_or_unbounded_cgroup_refuses(self):
        for limit in ("max", 64*MEM.GIB, 1000*MEM.GIB):
            rows = [sample() for _ in range(10)]
            for row in rows:
                row["cgroup"]["memory.max"] = limit
            with self.assertRaises(ValueError):
                MEM.summarize(rows)

    def test_incomplete_measurements_refuse(self):
        rows = [sample() for _ in range(10)]
        with self.assertRaises(ValueError):
            MEM.summarize(rows[:9])
        rows[-1]["cgroup"]["stat"].pop("anon")
        with self.assertRaises(ValueError):
            MEM.summarize(rows)
        rows = [sample() for _ in range(10)]
        for row in rows:
            row.pop("gpu")
        with self.assertRaises(ValueError):
            MEM.summarize(rows)

    def test_rss_anon_units_and_single_process_hwm(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / "status"
            status.write_text("Name:\tpython\nVmRSS:\t123 kB\nVmHWM:\t456 kB\nRssAnon:\t78 kB\n")
            self.assertEqual(MEM.process_status(status),
                             {"VmRSS": 123*1024, "VmHWM": 456*1024, "RssAnon": 78*1024})
        rows = [sample() for _ in range(10)]
        rows[-1]["processes"].append({"pid": 11, "VmRSS": 2*MEM.GIB,
                                     "VmHWM": 5*MEM.GIB, "RssAnon": MEM.GIB})
        result = MEM.summarize(rows)
        self.assertEqual(result["peak_process_tree_rss_bytes"], 24*MEM.GIB)
        self.assertEqual(result["peak_process_hwm_bytes"], 23*MEM.GIB)

    def test_process_tree_covers_children_of_all_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            for pid, children in ((10, "11"), (11, "")):
                directory = proc / str(pid)
                (directory / "task" / str(pid)).mkdir(parents=True)
                (directory / "status").write_text("VmRSS: 2 kB\nRssAnon: 1 kB\n")
                (directory / "task" / str(pid) / "children").write_text(children)
            self.assertEqual({p["pid"] for p in MEM.process_tree(10, proc)}, {10, 11})

    def test_cgroup_namespace_mount_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            mount = Path(tmp)
            (mount / "step").mkdir()
            (mount / "step/memory.current").write_text("42")
            info = f"1 0 0:1 /slurm/job123 {mount} ro - cgroup2 cgroup rw"
            self.assertEqual(MEM.resolve_cgroup("0::/slurm/job123/step", info,
                             mount / "absent"), mount / "step")

    def test_cgroup_v1_and_unsafe_membership_refuse(self):
        with self.assertRaises(RuntimeError):
            MEM.resolve_cgroup("5:memory:/slurm/job123", "")
        with self.assertRaises(ValueError):
            MEM.resolve_cgroup("0::/../../slurm", "")

    def test_job_parent_limit_with_unbounded_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step = root / "step"
            step.mkdir()
            (step / "memory.max").write_text("max")
            (root / "memory.max").write_text(str(180*MEM.GIB))
            self.assertEqual(MEM.allocation_cgroup(step), root)
            (step / "memory.max").write_text(str(64*MEM.GIB))
            with self.assertRaises(ValueError):
                MEM.allocation_cgroup(step)

    def test_enroot_root_namespace_uses_exact_host_job_not_batch_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            mount = Path(tmp)
            job = mount / "system.slice/slurmstepd.scope/job_123"
            job.mkdir(parents=True)
            (job / "memory.current").write_text("42")
            host = "0::/system.slice/slurmstepd.scope/job_123/step_batch"
            self.assertEqual(MEM.resolve_cgroup("0::/", "", mount, host, "123"), job)
            for wrong in ("124", "", "../../123"):
                with self.assertRaises(ValueError):
                    MEM.resolve_cgroup("0::/", "", mount, host, wrong)

    def test_prior_oom_event_in_same_allocation_refuses(self):
        rows = [sample() for _ in range(10)]
        for row in rows:
            row["cgroup"]["events"]["oom"] = 1
        self.assertFalse(MEM.summarize(rows)["memory_gate_passed"])

    def test_sampler_durable_jsonl_and_failure_reporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            monitor = MEM.Monitor(path, Path(tmp), interval=.001)
            with patch.object(monitor, "sample", return_value=sample()):
                monitor.start()
                monitor.finish()
            self.assertGreaterEqual(len(path.read_text().splitlines()), 2)
            json.loads(path.read_text().splitlines()[0])
            failing = MEM.Monitor(path, Path(tmp), interval=.001)
            with patch.object(failing, "sample", return_value=sample()):
                failing.start()
                with self.assertRaises(RuntimeError):
                    failing.finish()

    def test_existing_prediction_refuses_without_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proof = root / "receipts/epoch400/pair149.json"
            proof.parent.mkdir(parents=True)
            proof.write_text("{}")
            from unittest.mock import Mock
            driver = Mock()
            driver.safe_root.return_value = root
            case = {"kspace_shape": MEM.EXPECTED_RAW}
            driver.verify.return_value = ({}, {"cases": [case]*34,
                "pairs": [{"index": 149, "case_index": 33, "R": 10}]})
            with patch.object(MEM, "load_driver", return_value=driver):
                with self.assertRaises(FileExistsError):
                    MEM.profile(root, root)
            driver.infer_pairs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
