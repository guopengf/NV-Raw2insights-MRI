import importlib.util
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

P = Path(__file__).resolve().parents[1]/"scripts/tools/run_unified_checkpoint_validation_e800.py"
S = importlib.util.spec_from_file_location("e800",P)
e = importlib.util.module_from_spec(S); S.loader.exec_module(e)


class Epoch800Test(unittest.TestCase):
    def test_cpu_launcher_discards_inherited_absolute_binding_mask(self):
        launcher=P.parents[1]/"slurm/run_unified_checkpoint_validation_e800_cpu.slurm"
        script='export VALIDATION_RUN_REL=not-a-real-run VALIDATION_STAGE=prepare; export SLURM_CPU_BIND=mask_cpu:0xffff0000 SLURM_CPU_BIND_TYPE=mask_cpu SLURM_CPU_BIND_LIST=0xffff0000 SLURM_CPU_BIND_VERBOSE=verbose; srun(){ printf "%s\\n" "${SLURM_CPU_BIND_TYPE-unset}" "${SLURM_CPU_BIND_LIST-unset}" "$1"; }; source "$1"'
        result=subprocess.check_output(["bash","-c",script,"test",str(launcher)],text=True)
        self.assertEqual(result.splitlines(),["unset","unset","--cpu-bind=cores"])

    def plan(self):
        cases = [dict(index=i,anatomy="Aorta",center="C",vendor="V",patient=str(i),
            rel_dir=f"case{i}",kspace_shape=[4,23,10,20,108,84],image_shape=[4,23,20,108,84],image_bytes=1) for i in range(63)]
        pairs = [dict(index=i,case_index=i%63,R=10,rel_path=f"p{i}") for i in range(259)]
        return dict(cases=cases,pairs=pairs)

    def test_private_epoch_scope_preserves_original_driver(self):
        original = importlib.util.spec_from_file_location("original",e.CORE)
        m = importlib.util.module_from_spec(original); original.loader.exec_module(m)
        self.assertEqual(m.EPOCHS,(400,500,600))
        self.assertEqual(e.v.EPOCHS,(800,))
        self.assertEqual(e.v.sha(e.CORE),e.CORE_SHA)

    def test_requested_checkpoint_hash_is_fixed(self):
        self.assertEqual(e.CHECKPOINT_SHA,"017384e1f40bfae94a4c22aabd94d0988e05412d13059c4c0c6cb6c9dd86172a")

    def test_disjoint_roots(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            base = repo/e.BASE_REL
            self.assertEqual(e.base_root(repo,repo/"outputs/4dflow/checkpoint_validation/extension"),base)
            for root in (base,base/"child",base.parent):
                with self.assertRaises(ValueError):e.base_root(repo,root)

    def test_existing_epoch700_output_is_protected(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d); old = repo/e.PROTECTED_E700_REL
            for root in (old,old/"child",old.parent):
                with self.assertRaises(ValueError):e.base_root(repo,root)

    def test_gpu_launcher_discards_inherited_absolute_binding_mask(self):
        launcher=P.parents[1]/"slurm/run_unified_checkpoint_validation_e800_gpu.slurm"
        script='export VALIDATION_RUN_REL=not-a-real-run VALIDATION_STAGE=prepare; export SLURM_CPU_BIND=mask_cpu:0xffff0000 SLURM_CPU_BIND_TYPE=mask_cpu SLURM_CPU_BIND_LIST=0xffff0000 SLURM_CPU_BIND_VERBOSE=verbose; srun(){ printf "%s\\n" "${SLURM_CPU_BIND_TYPE-unset}" "${SLURM_CPU_BIND_LIST-unset}" "$1"; }; source "$1"'
        result=subprocess.check_output(["bash","-c",script,"test",str(launcher)],text=True)
        self.assertEqual(result.splitlines(),["unset","unset","--cpu-bind=cores"])

    def test_exact_split_identity(self):
        a = self.plan(); b = self.plan(); e.same_plan(a,b)
        b["cases"][0]["patient"]="training-patient"
        with self.assertRaises(ValueError):e.same_plan(a,b)

    def test_acceleration_coverage_drift_refused(self):
        a = self.plan(); b = self.plan(); b["pairs"][1]["R"]=50
        with self.assertRaises(ValueError):e.same_plan(a,b)
        b = self.plan(); b["pairs"].pop()
        with self.assertRaises(ValueError):e.same_plan(b,a)

    def test_single_epoch_array_isolated_subprocesses(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            e.v.write(root/"shards.json",dict(shards=[dict(indices=[1,2]),dict(indices=[3])]))
            with patch.object(e,"require_base_complete"),patch.object(e.v,"infer_pairs") as infer:
                e.infer_array(root,root,0)
            self.assertEqual([c.args for c in infer.call_args_list],[
                (root,root,800,[1],"full_task000_p001"),(root,root,800,[2],"full_task000_p002")])

    def test_single_epoch_array_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); e.v.write(root/"shards.json",dict(shards=[dict(indices=[0])]))
            with patch.object(e,"require_base_complete"),patch.object(e.v,"infer_pairs") as infer:
                for index in (None,-1,1,2,3):
                    with self.assertRaises(IndexError):e.infer_array(root,root,index)
                infer.assert_not_called()

    def test_no_gpu_stage_before_existing_validation_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(e,"require_base_complete",side_effect=ValueError("incomplete")),patch.object(e.v,"infer_pairs") as infer:
                for action in (lambda:e.pilot(root,root),lambda:e.infer_array(root,root,0)):
                    with self.assertRaises(ValueError):action()
                infer.assert_not_called()

    def test_duplicate_task_lock_not_bypassed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); e.v.write(root/"shards.json",dict(shards=[dict(indices=[0])]))
            with e.v.lock(root,"inference_task000"),patch.object(e,"require_base_complete"),patch.object(e.v,"infer_pairs") as infer:
                with self.assertRaises(BlockingIOError):e.infer_array(root,root,0)
                infer.assert_not_called()

    def test_new_pilot_reuses_original_single_case_path(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); plan=self.plan(); plan["cases"][62]["image_bytes"]=2
            for index in (0,62):
                e.v.write(root/"work"/f"pilot_e800_p{index}"/"epoch800/receipt.json",dict(status="complete",epoch=800,pair_indices=[index],wall_seconds=10,samples=1932))
            with patch.object(e,"require_base_complete",return_value=(plan,root,{})),patch.object(e.v,"infer_pairs") as infer:
                e.pilot(root,root)
            self.assertEqual([c.args[2:4] for c in infer.call_args_list],[(800,[0]),(800,[62])])
            self.assertEqual(len(e.v.read(root/"pilot.json")["measurements"]),2)

    def test_shards_keep_conservative_old_rate_and_9000_limit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"extension"; base=Path(d)/"base"; plan=self.plan()
            e.v.write(base/"shards.json",dict(seconds_per_sample=1.5))
            e.v.write(root/"pilot.json",dict(status="complete",measurements=[dict(pair_index=0,wall_seconds=100),dict(pair_index=62,wall_seconds=200)]))
            with patch.object(e,"verify",return_value=({},plan,base)):e.make_shards(root,root)
            r=e.v.read(root/"shards.json")
            self.assertEqual(r["epochs"],[800]); self.assertEqual(r["seconds_per_sample"],1.5)
            self.assertEqual(r["total_array_tasks"],len(r["shards"]))
            self.assertEqual(sorted(i for s in r["shards"] for i in s["indices"]),list(range(259)))
            self.assertTrue(all(s["estimated_seconds"]<=9000 for s in r["shards"]))

    def test_new_slow_pilot_increases_rate_with_same_slack(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"extension"; base=Path(d)/"base"; plan=self.plan()
            e.v.write(base/"shards.json",dict(seconds_per_sample=.1))
            e.v.write(root/"pilot.json",dict(status="complete",measurements=[dict(pair_index=0,wall_seconds=1932),dict(pair_index=62,wall_seconds=1932)]))
            with patch.object(e,"verify",return_value=({},plan,base)):e.make_shards(root,root)
            self.assertEqual(e.v.read(root/"shards.json")["seconds_per_sample"],1.5)

    def test_record_job_rejects_duplicate_stage_with_other_id(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); e.v.record_job(root,"prepare","123"); e.v.record_job(root,"prepare","123")
            with self.assertRaises(ValueError):e.v.record_job(root,"prepare","124")
            self.assertEqual(len(e.v.read(root/"jobs.json")),1)

    def test_nonfinite_score_and_duplicate_coverage_refused(self):
        row=dict(rel_path="p",comments="",**{m:1 for m in e.v.METRICS})
        e.v.validate_rows([row],{"p"})
        with self.assertRaises(ValueError):e.v.validate_rows([row,row],{"p"})
        for metric in e.v.METRICS:
            with self.assertRaises(ValueError):e.v.validate_rows([dict(row,**{metric:"nan"})],{"p"})


if __name__=="__main__":unittest.main()
