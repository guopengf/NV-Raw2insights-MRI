import importlib.util
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
P=Path(__file__).resolve().parents[1]/"scripts/tools/run_unified_checkpoint_validation.py"
S=importlib.util.spec_from_file_location("validation",P)
v=importlib.util.module_from_spec(S);S.loader.exec_module(v)

def concurrent_array_probe(root,index,ready,release,results):
    try:
        def started(*args):
            ready.put(index)
            if not release.wait(45):raise TimeoutError("Concurrent lock probe timed out")
        with patch.object(v,"infer_pairs",side_effect=started):
            v.infer_array(Path(root),Path(root),index)
        results.put((index,"ok"))
    except BaseException as error:
        results.put((index,type(error).__name__))

class Tensor:
    def __init__(self,shape):self.shape=shape

class UnifiedValidationTest(unittest.TestCase):
    def test_checkpoint_identity_across_mount_namespaces(self):
        self.assertTrue(v.same_checkpoint_identity([35,100,200,300,400],[48,100,200,300,400]))
        self.assertFalse(v.same_checkpoint_identity([35,101,200,300,400],[48,100,200,300,400]))
        self.assertFalse(v.same_checkpoint_identity([35,100,201,300,400],[48,100,200,300,400]))
        self.assertFalse(v.same_checkpoint_identity([35,100,200,301,400],[48,100,200,300,400]))
    def test_protected_roots(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);v.safe_root(repo,repo/"outputs/4dflow/checkpoint_validation/run")
            for p in (repo,repo/"outputs/4dflow/4dflow_joint_encoding/run",repo/"outputs/4dflow/checkpoint_validation"):
                with self.assertRaises(ValueError):v.safe_root(repo,p)
    def test_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);a=repo/"outputs/4dflow/checkpoint_validation";a.mkdir(parents=True)
            (a/"escape").symlink_to(repo)
            with self.assertRaises(ValueError):v.safe_root(repo,a/"escape")
    def test_minimum_joint_batch(self):
        source=dict(phase3=dict(recon_mode="slab",num_slices=3,enable_vaa=False,
            joint_encoding=dict(enabled=True,mode="channel",count=4,order=[0,1,2,3])),
            four_dflow_augmentation=dict(enabled=True))
        result=v.effective_config(source)
        self.assertEqual(result["batch_size"],4);self.assertEqual(result["num_workers"],0)
        self.assertTrue(source["four_dflow_augmentation"]["enabled"])
    def test_wrong_order(self):
        source=dict(phase3=dict(joint_encoding=dict(enabled=True,mode="channel",count=4,order=[3,2,1,0])))
        with self.assertRaises(ValueError):v.effective_config(source)
    def test_targetless(self):
        p=v.input_payload(dict(source_dir="/data/val/case"),20)
        self.assertEqual(p["target_kspace"],p["kspace"]);self.assertTrue(p["targetless"])
        self.assertNotIn("kdata_full",str(p));self.assertEqual(p["encoding_indices"],[0,1,2,3])
    def test_strict_shapes(self):
        v.strict_state({"module.a":Tensor((2,3))},{"a":(2,3)})
        for state in ({},{"a":Tensor((3,2))},{"a":Tensor((2,3)),"b":Tensor((1,))}):
            with self.assertRaises(ValueError):v.strict_state(state,{"a":(2,3)})
    def test_key_collision(self):
        with self.assertRaises(ValueError):v.strict_state({"a":Tensor((1,)),"module.a":Tensor((1,))},{"a":(1,)})
    def test_inference_center_axis(self):
        self.assertEqual(v.inference_center_samples({"kspace_shape":[4,23,10,20,108,84]}),1932)
        self.assertEqual(v.inference_center_samples({"kspace_shape":[4,12,8,88,198,176]}),2112)
        for shape in ([4,2,3],[3,23,10,20,108,84],[4,0,10,20,108,84]):
            with self.assertRaises(ValueError):v.inference_center_samples({"kspace_shape":shape})
    def test_audited_amendment_requires_reason(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):v.amend_wrapper_pin(Path(d),Path(d),"")
    def test_pilot_workload_is_rederived_before_partition(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            plan=dict(cases=[dict(kspace_shape=[4,23,10,20,108,84],samples=460)],
                      pairs=[dict(index=0,case_index=0,R=10)])
            v.write(root/"pilot.json",dict(status="complete",measurements=[dict(
                epoch=400,pair_index=0,wall_seconds=100,samples=460)]))
            with patch.object(v,"verify",return_value=({},plan)):v.make_shards(root,root)
            result=v.read(root/"shards.json")
            self.assertAlmostEqual(result["seconds_per_sample"],100/1932*1.5)
            self.assertEqual(result["pilot_timings"][0]["reported_samples"],460)
            self.assertEqual(result["pilot_timings"][0]["inference_center_samples"],1932)
            self.assertAlmostEqual(result["shards"][0]["estimated_seconds"],180)
    def test_shard_coverage(self):
        cases=[dict(samples=10),dict(samples=20)]
        pairs=[dict(index=i,case_index=i%2,R=10+i) for i in range(7)]
        shards=v.partition(pairs,cases,1,100)
        self.assertEqual(sorted(i for s in shards for i in s["indices"]),list(range(7)))
        self.assertTrue(all(s["estimated_seconds"]<=100 for s in shards))
    def test_oversized_case(self):
        with self.assertRaises(ValueError):v.partition([dict(index=0,case_index=0,R=10)],[dict(samples=1000)],1,100)
    def test_array_isolates_each_case_with_original_single_case_path(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            v.write(root/"shards.json",dict(shards=[dict(indices=[0,1]),dict(indices=[2])]))
            with patch.object(v,"infer_pairs") as infer:
                v.infer_array(root,root,0)
            self.assertEqual([call.args for call in infer.call_args_list],[
                (root,root,400,[0],"full_task000_p000"),
                (root,root,400,[1],"full_task000_p001")])
            for task,epoch in ((3,500),(5,600)):
                with patch.object(v,"infer_pairs") as infer:
                    v.infer_array(root,root,task)
                infer.assert_called_once_with(root,root,epoch,[2],f"full_task{task:03d}_p002")
    def test_array_outside_plan_never_starts_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            v.write(root/"shards.json",dict(shards=[dict(indices=[0])]))
            with patch.object(v,"infer_pairs") as infer:
                for task in (-1,3):
                    with self.assertRaises(IndexError):v.infer_array(root,root,task)
                infer.assert_not_called()
    def test_sixteen_distinct_array_tasks_can_overlap_in_real_processes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            v.write(root/"shards.json",dict(shards=[dict(indices=[i]) for i in range(16)]))
            context=multiprocessing.get_context("spawn")
            ready=context.Queue();results=context.Queue();release=context.Event()
            processes=[context.Process(target=concurrent_array_probe,args=(d,i,ready,release,results)) for i in range(16)]
            try:
                for process in processes:process.start()
                self.assertEqual(sorted(ready.get(timeout=35) for _ in processes),list(range(16)))
                release.set()
                self.assertEqual(sorted(results.get(timeout=15) for _ in processes),[(i,"ok") for i in range(16)])
            finally:
                release.set()
                for process in processes:
                    if process.pid is not None:
                        process.join(timeout=5)
                        if process.is_alive():process.terminate();process.join(timeout=5)
                ready.close();results.close()
            self.assertTrue(all(process.exitcode==0 for process in processes))
    def test_duplicate_task_still_refused_before_any_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);v.write(root/"shards.json",dict(shards=[dict(indices=[0])]))
            with v.lock(root,"inference",shared=True), v.lock(root,"inference_task000"):
                with patch.object(v,"infer_pairs") as infer:
                    with self.assertRaises(BlockingIOError):v.infer_array(root,root,0)
                    infer.assert_not_called()
    def test_exclusive_pilot_and_shared_array_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);v.write(root/"shards.json",dict(shards=[dict(indices=[0])]))
            with v.lock(root,"inference"):
                with patch.object(v,"infer_pairs") as infer:
                    with self.assertRaises(BlockingIOError):v.infer_array(root,root,0)
                    infer.assert_not_called()
            with v.lock(root,"inference",shared=True):
                with self.assertRaises(BlockingIOError):
                    with v.lock(root,"inference"):self.fail("Exclusive pilot acquired shared array lock")
    def test_array_locks_released_after_inference_exception(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);v.write(root/"shards.json",dict(shards=[dict(indices=[0])]))
            with patch.object(v,"infer_pairs",side_effect=RuntimeError("probe")):
                with self.assertRaises(RuntimeError):v.infer_array(root,root,0)
            with patch.object(v,"infer_pairs") as infer:v.infer_array(root,root,0)
            infer.assert_called_once()
    def test_recovery_uses_separate_attempt_work_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);v.write(root/"shards.json",dict(shards=[dict(indices=[0,1])]))
            with patch.object(v,"infer_pairs") as infer:
                v.infer_array(root,root,0,"recovery_array123")
            self.assertEqual([c.args[-1] for c in infer.call_args_list],[
                "recovery_array123_task000_p000","recovery_array123_task000_p001"])
    def test_unsafe_recovery_label_never_starts_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);v.write(root/"shards.json",dict(shards=[dict(indices=[0])]))
            with patch.object(v,"infer_pairs") as infer:
                for label in ("","../original","a/b","a b","."):
                    with self.assertRaises(ValueError):v.infer_array(root,root,0,label)
                infer.assert_not_called()
    def test_scores(self):
        row=dict(rel_path="case",comments="",**{m:"1" for m in v.METRICS})
        v.validate_rows([row],{"case"})
        for rows in ([row,row],[dict(row,nRMSE="nan")],[dict(row,comments="error")]):
            with self.assertRaises(ValueError):v.validate_rows(rows,{"case"})
    def test_missing_scores(self):
        with self.assertRaises(ValueError):v.validate_rows([],{"missing"})
    def test_record_jobs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);v.record_job(root,"pilot","123");v.record_job(root,"pilot","123")
            self.assertEqual(len(v.read(root/"jobs.json")),1)
            with self.assertRaises(ValueError):v.record_job(root,"pilot","124")

if __name__=="__main__":unittest.main()
