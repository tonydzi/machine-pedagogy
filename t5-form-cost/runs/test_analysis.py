"""Unit-only FAKE 54-result assay, held entirely in memory; no scientific JSON writes.

The full driver freeze receipt is mocked here; actual driver.valid_result and
all_complete run against fake bytes. These tests prove wrapper conditions, not
acceptance of a real frozen experiment. No Torch, training, tasks or inference.
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import analyze


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def fake_hash(label):
    return hashlib.sha256(label.encode()).hexdigest()


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(Path.cwd().anchor) / "__T5_FAKE_UNIT_ONLY__"
        cls.manifest = cls.root / "manifest.json"
        cls.out = cls.root / "summary.json"
        cls.output_dir = cls.root / "results"
        cls.fs_base = {}
        cfg = dict(steps_a=200, steps_b=200, steps_c=150, threads=1, batch=64,
                   width=80, layers=2, heads=4, lr=.001, checkpoint=25,
                   train_n=512, dev_n=128, test_n=256, comp_n=128)
        m = dict(phase="confirmatory", seeds=analyze.SEEDS, arms=analyze.driver.ARMS,
                 config=cfg, Machine="FAKE-UNIT-TEST", output_dir=str(cls.output_dir),
                 t5=str(cls.root / "t5.py"), gate_receipt=str(cls.root / "receipt.json"),
                 data_hashes=dict(path=str(cls.root / "frozen.json")))
        cls.fs_base[m["t5"]] = b"FAKE CORE NEVER EXECUTED"
        cls.fs_base[m["gate_receipt"]] = b"FAKE FULL GATE MOCKED BY TEST"
        cls.fs_base[str(Path(analyze.__file__))] = b"FAKE ANALYZER PROVENANCE"
        m["code_sha256"] = hashlib.sha256(cls.fs_base[m["t5"]]).hexdigest()
        data_by_seed, frozen_rows = {}, []
        for seed in analyze.SEEDS:
            phase = {split: [[i % 8, (i // 8 + i) % 8, 0, 0] for i in range(cfg[split + "_n"])]
                     for split in ("train", "dev", "test", "comp")}
            data = dict(schema="t5-data/v1", seed=seed, layout={"FAKE": seed},
                        config=cfg, data=[copy.deepcopy(phase) for _ in range(3)], rng="FAKE", Machine=m["Machine"])
            data["data"][2]["dev"] = [[0, 1, 0, 0], [2, 1, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0]] * 32
            data["data"][2]["comp"] = [[0, 3, 0, 0], [3, 0, 0, 0], [1, 3, 0, 0], [2, 2, 0, 0]] * 32
            data["split_sha256"] = analyze.digest(data["data"])
            data_by_seed[seed] = data
            frozen_rows.append(dict(seed=seed, layout=data["layout"],
                                    data_manifest_sha256=analyze.digest(data),
                                    split_sha256=data["split_sha256"],
                                    phases=[{s: dict(n=len(records), sha256=analyze.digest(records))
                                             for s, records in p.items()} for p in data["data"]]))
        frozen = dict(schema="t5-frozen-data-hashes/v1", seeds=analyze.SEEDS,
                      code_sha256=m["code_sha256"], config=cfg,
                      config_sha256=analyze.digest(cfg), rows=frozen_rows)
        cls.fs_base[m["data_hashes"]["path"]] = encoded(frozen)
        m["data_hashes"]["sha256"] = hashlib.sha256(encoded(frozen)).hexdigest()
        cls.fs_base[str(cls.manifest)] = encoded(m)
        m["_manifest_path"] = str(cls.manifest)
        m["_manifest_sha256"] = hashlib.sha256(cls.fs_base[str(cls.manifest)]).hexdigest()
        counts = {"probe_dev": (48000, 3200, 25, 0), "train_A": (192000, 12800, 200, 200),
                  "retention_pre_test": (3840, 256, 2, 0), "acquisition_native_test": (3840, 256, 2, 0),
                  "train_B": (192000, 12800, 200, 200), "retention_post_test": (3840, 256, 2, 0),
                  "retention_native_post_test": (3840, 256, 2, 0), "h1_test": (3840, 256, 2, 0),
                  "composition_pre_test": (1920, 128, 1, 0), "secondary_matrix": (15360, 1024, 8, 0),
                  "train_C": (144000, 9600, 150, 150), "composition_test": (1920, 128, 1, 0)}
        categories = {key: dict(zip(analyze.TOTAL, values)) for key, values in counts.items()}
        for seed, arm, path in analyze.driver.runs(m):
            common, native = fake_hash(f"common-{seed}"), fake_hash(f"native-{seed}-{arm}")
            row = dict(schema="t5-result/v1", status="complete", seed=seed, arm=arm,
                       code_sha256=m["code_sha256"], manifest_sha256=m["_manifest_sha256"], config=cfg,
                       device="cpu", Machine=m["Machine"], instrument_revision="native-retention-v2",
                       data_manifest=data_by_seed[seed], data_manifest_sha256=analyze.digest(data_by_seed[seed]),
                       training_semantic_sha256=[fake_hash(f"semantic-{seed}-{i}") for i in range(3)],
                       c_training_inputs_sha256=fake_hash(f"c-{seed}"),
                       probe_hashes=dict(a_common_pre=common, a_common_post=common,
                                         a_native_pre=native, a_native_post=native),
                       native_probe_audit=dict(weights_before=fake_hash("weights"),
                                               weights_after=fake_hash("weights"), torch_rng_unchanged=True),
                       metrics=dict(a_acquisition_native=.96, a_native_post_b=.85, retention_native_delta=-.11,
                                    a_pre_b=.35, a_post_b=.30, retention_delta=-.05, h1_b_post=.5, c_auc=.6,
                                    c_comp_pre=.4, c_comp=.6, c_comp_gain=.2),
                       secondary_matrix={a: .9 for a in analyze.driver.ARMS},
                       curves=[dict(phase=p, step=s, accuracy=(.3 + .004 * s if p == 2 else .6),
                                    inputs_sha256=fake_hash(f"dev-{seed}-{p}"))
                               for p, end in enumerate((200, 200, 150))
                               for s in range(0, end + 1, 25)],
                       accounting=dict(total=analyze.TOTAL, categories=categories, padding_tokens=0,
                                       restoration_steps=0, wall_s=1., cpu_process_s=.8, preprocessing_wall_s=.1))
            cls.fs_base[str(path)] = encoded(row)
        cls.m_base = m

    def setUp(self):
        self.fs = self.fs_base.copy()
        self.m = copy.deepcopy(self.m_base)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(analyze.driver, "read_manifest", side_effect=lambda _: self.m))
        self.stack.enter_context(patch.object(Path, "read_bytes", lambda path: self.fs[str(path)]))
        self.stack.enter_context(patch.object(analyze.driver, "read_json", side_effect=lambda p: json.loads(self.fs[str(p)])))
        self.stack.enter_context(patch.object(Path, "iterdir", lambda path: iter(
            [Path(p) for p in self.fs if Path(p).parent == path])))
        self.stack.enter_context(patch.object(Path, "is_file", lambda path: str(path) in self.fs))
        self.stack.enter_context(patch.object(Path, "is_symlink", return_value=False))
        self.load = self.stack.enter_context(patch.object(analyze, "load_core"))
        self.publish = self.stack.enter_context(patch.object(analyze, "publish", return_value=True))
        self.mkdir = self.stack.enter_context(patch.object(Path, "mkdir"))
        self.popen = self.stack.enter_context(patch("subprocess.Popen"))

    def path(self, seed=20, arm="fixed"):
        return self.output_dir / f"seed{seed}_{arm}.json"

    def mutate(self, change, seed=20, arm="fixed"):
        path = self.path(seed, arm)
        row = json.loads(self.fs[str(path)])
        change(row)
        self.fs[str(path)] = encoded(row)

    def run_cli(self, check=True):
        args = ["--manifest", str(self.manifest), "--out", str(self.out)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
            rc = analyze.main(args + (["--check"] if check else []))
        return rc, error.getvalue()

    def blocked(self):
        rc, error = self.run_cli()
        self.assertEqual(rc, 1, error)
        self.assertIn("BLOCKED", error)
        self.load.assert_not_called()
        self.publish.assert_not_called()
        self.mkdir.assert_not_called()
        self.popen.assert_not_called()

    def fake_summary(self, n=18):
        runs = [dict(seed=s, arm=a, metrics=json.loads(self.fs[str(p)])["metrics"])
                for s, a, p in analyze.driver.runs(self.m)]
        value = dict(n_paired_seeds=n, differences=[.01] * n, runs=runs,
                     ci95=[-.02, .04], p_two_sided=.4, rough_normal_n_for_3pp_80pct=13,
                     power_caveat="pilot n=3 is unstable; not adequate power evidence")
        self.load.return_value = SimpleNamespace(summarize=lambda _: print(json.dumps(value)))
        return value

    def test_full54_check_is_read_only_and_no_inference(self):
        before = self.fs.copy()
        self.assertEqual(self.run_cli(), (0, ""))
        self.assertEqual(before, self.fs)
        self.load.assert_not_called()
        self.publish.assert_not_called()
        self.mkdir.assert_not_called()
        self.popen.assert_not_called()

    def test_full54_summary_preserves_inference_and_no_exclusions(self):
        expected = self.fake_summary()
        self.assertEqual(self.run_cli(False), (0, ""))
        report = self.publish.call_args.args[1]
        for key in ("ci95", "p_two_sided", "differences"):
            self.assertEqual(report[key], expected[key])
        self.assertEqual(report["no_exclusion"]["included_seeds"], analyze.SEEDS)
        self.assertEqual(report["no_exclusion"]["excluded_seeds"], [])
        self.assertFalse(report["feasibility"]["assay_limited"])
        self.assertAlmostEqual(report["feasibility"]["mean_B_majority"], .125)
        self.assertEqual(report["feasibility"]["H1_floor_threshold"], .25)
        self.assertNotIn("native_retention_contrast", report)
        self.assertNotIn("rough_normal_n_for_3pp_80pct", report)
        self.assertEqual(report["power_caveat"], "Finite 18-seed pilot-scale series; planned power conditional on 3 dev seeds; no post-hoc n adaptation")
        per_run = report["feasibility"]["per_run"]
        self.assertEqual(len(per_run), 54)
        self.assertEqual({(r["seed"], r["arm"]) for r in per_run},
                         {(s, a) for s in analyze.SEEDS for a in analyze.driver.ARMS})
        for row in per_run:
            self.assertAlmostEqual(row["C_dev_initial"], .3)
            self.assertAlmostEqual(row["C_dev_final"], .9)
            self.assertEqual(row["C_dev_baselines"], dict(n=128, majority=.5, oracle_copy_A=.75, oracle_copy_B=.75))
            self.assertEqual(row["C_comp_baselines"], dict(n=128, majority=.75, oracle_copy_A=.5, oracle_copy_B=.75))
        for row in report["runs"]:
            self.assertEqual([row["metrics"][k] for k in ("c_comp_pre", "c_comp", "c_comp_gain")], [.4, .6, .2])

    def test_addendum_manual_oracle_copy_example_counts_tie_for_both(self):
        records = [(0, 3, 0, 0), (7, 1, 0, 0), (2, 2, 0, 0), (3, 7, 0, 0)]
        self.assertEqual(analyze.c_baselines(records),
                         dict(n=4, majority=.5, oracle_copy_A=.5, oracle_copy_B=.75))

    def test_missing_h3_report_field_blocks_before_summary(self):
        original = analyze.feasibility
        def incomplete(rows):
            diagnostic = original(rows)
            diagnostic["per_run"][0].pop("C_comp_baselines")
            return diagnostic
        with patch.object(analyze, "feasibility", side_effect=incomplete):
            self.assertEqual(self.run_cli(False)[0], 1)
        self.load.assert_not_called()
        self.publish.assert_not_called()

    def test_n17_incomplete_and_extra_file_block_before_summary(self):
        for mode in ("n17", "missing", "extra"):
            with self.subTest(mode=mode):
                self.fs = self.fs_base.copy()
                if mode == "extra":
                    self.fs[str(self.output_dir / "unexpected.json")] = b"{}"
                else:
                    for arm in (analyze.driver.ARMS if mode == "n17" else ["fixed"]):
                        self.fs.pop(str(self.path(37, arm)))
                self.blocked()

    def test_wrong_phase_seeds_status_config_or_sha_block(self):
        for mode in ("phase", "seeds", "status", "config", "code", "manifest", "source"):
            with self.subTest(mode=mode):
                self.fs, self.m = self.fs_base.copy(), copy.deepcopy(self.m_base)
                if mode == "phase": self.m["phase"] = "pilot-dev"
                elif mode == "seeds": self.m["seeds"] = list(range(20, 37))
                elif mode == "source": self.fs[self.m["t5"]] += b"FAKE SOURCE CHANGED"
                else:
                    def change(row):
                        if mode == "config": row["config"]["steps_a"] = 201
                        else: row[{"status": "status", "code": "code_sha256", "manifest": "manifest_sha256"}[mode]] = "WRONG"
                    self.mutate(change)
                self.blocked()

    def test_frozen_raw_sha_and_embedded_data_hash_block(self):
        for mode in ("raw", "declared", "embedded"):
            with self.subTest(mode=mode):
                self.fs = self.fs_base.copy()
                if mode == "raw": self.fs[self.m["data_hashes"]["path"]] += b" "
                elif mode == "declared": self.mutate(lambda r: r.update(data_manifest_sha256="0" * 64))
                else: self.mutate(lambda r: r["data_manifest"]["data"][1]["test"][0].__setitem__(1, 7))
                self.blocked()

    def test_actual_pairs_semantics_and_c_inputs_block(self):
        for mode in ("common_pair", "native_pair", "common_cross_arm", "semantic", "c", "hash_shape",
                     "c_dev", "dev_hash_shape"):
            with self.subTest(mode=mode):
                self.fs = self.fs_base.copy()
                def change(row):
                    if mode == "common_pair": row["probe_hashes"]["a_common_post"] = "0" * 64
                    elif mode == "native_pair": row["probe_hashes"]["a_native_post"] = "0" * 64
                    elif mode == "common_cross_arm":
                        row["probe_hashes"].update(a_common_pre="0" * 64, a_common_post="0" * 64)
                    elif mode == "semantic": row["training_semantic_sha256"][0] = "0" * 64
                    elif mode == "c": row["c_training_inputs_sha256"] = "0" * 64
                    elif mode == "c_dev": row["curves"][-1]["inputs_sha256"] = "0" * 64
                    elif mode == "dev_hash_shape": row["curves"][-1]["inputs_sha256"] = ""
                    else: row["probe_hashes"]["a_native_pre"] = ""
                self.mutate(change)
                self.blocked()

    def test_total_and_additional_native_probe_budget_block(self):
        for mode in ("total", "native_probe", "sum"):
            with self.subTest(mode=mode):
                self.fs = self.fs_base.copy()
                def change(row):
                    ledger = row["accounting"]
                    if mode == "total": ledger["total"]["tokens"] -= 1
                    elif mode == "native_probe": ledger["categories"]["retention_native_post_test"]["backward_calls"] = 1
                    else: ledger["categories"]["train_A"]["examples"] -= 1
                self.mutate(change)
                self.blocked()

    def test_native_weights_rng_and_nonfinite_probability_block(self):
        for mode in ("weights", "rng", "nan", "native_b", "native_post"):
            with self.subTest(mode=mode):
                self.fs = self.fs_base.copy()
                def change(row):
                    if mode == "weights": row["native_probe_audit"]["weights_after"] = "0" * 64
                    elif mode == "rng": row["native_probe_audit"]["torch_rng_unchanged"] = 1
                    elif mode == "nan": row["metrics"]["a_acquisition_native"] = float("nan")
                    elif mode == "native_b": row["secondary_matrix"]["fixed"] = 1.1
                    else: row["metrics"]["a_native_post_b"] = -.1
                self.mutate(change)
                self.blocked()

    def test_composition_probabilities_gain_and_curve_auc_block(self):
        for mode in ("pre", "post", "gain", "step", "phase", "auc", "curve_nan"):
            with self.subTest(mode=mode):
                self.fs = self.fs_base.copy()
                def change(row):
                    if mode == "pre": row["metrics"]["c_comp_pre"] = -.1
                    elif mode == "post": row["metrics"]["c_comp"] = 1.1
                    elif mode == "gain": row["metrics"]["c_comp_gain"] = 99
                    elif mode == "step": row["curves"][-1]["step"] = 149
                    elif mode == "phase": row["curves"][-1]["phase"] = 1
                    elif mode == "auc": row["metrics"]["c_auc"] = .7
                    else: row["curves"][-1]["accuracy"] = float("nan")
                self.mutate(change)
                self.blocked()

    def test_core_executes_only_the_bytes_bound_to_manifest_sha(self):
        path = Path(self.m["t5"])
        self.fs[str(path)] = b"FAKE_BOUND_VALUE = 7\n"
        sha = hashlib.sha256(self.fs[str(path)]).hexdigest()
        module = ORIGINAL_LOAD_CORE(path, sha)
        self.assertEqual(module.FAKE_BOUND_VALUE, 7)
        self.fs[str(path)] = b"FAKE_BOUND_VALUE = 8\n"
        with self.assertRaisesRegex(ValueError, "executed core bytes SHA"):
            ORIGINAL_LOAD_CORE(path, sha)

    def test_feasibility_flags_keep_all18_ci_and_p(self):
        for mode, flag in (("apre", "any_native_A_pre_below_090"), ("bnative", "any_arm_mean_B_native_below_080"),
                           ("floor", "H1_global_floor"), ("ceiling", "all_arm_mean_H1_at_least_095")):
            with self.subTest(mode=mode):
                self.fs = self.fs_base.copy()
                for seed, arm, _ in analyze.driver.runs(self.m):
                    def change(row):
                        if mode == "apre": row["metrics"].update(a_acquisition_native=.89, retention_native_delta=-.04)
                        elif mode == "bnative": row["secondary_matrix"][arm] = .79
                        else: row["metrics"]["h1_b_post"] = .2 if mode == "floor" else .96
                    self.mutate(change, seed, arm)
                self.fake_summary()
                self.assertEqual(self.run_cli(False), (0, ""))
                report = self.publish.call_args.args[1]
                self.assertTrue(report["feasibility"]["assay_limited"])
                self.assertTrue(report["feasibility"]["flags"][flag])
                self.assertEqual(report["n_paired_seeds"], 18)
                self.assertEqual(report["p_two_sided"], .4)
                self.assertEqual(report["ci95"], [-.02, .04])

    def test_core_summary_n17_cannot_publish(self):
        self.fake_summary(17)
        self.assertEqual(self.run_cli(False)[0], 1)
        self.publish.assert_not_called()

    def test_exact_thresholds_and_weak_run_diagnostics_keep_all54(self):
        for seed, arm, _ in analyze.driver.runs(self.m):
            def boundary(row):
                row["metrics"].update(a_acquisition_native=.9, retention_native_delta=-.05,
                                      a_pre_b=.8, a_post_b=.7, retention_delta=-.1, h1_b_post=.25)
                row["secondary_matrix"][arm] = .8
            self.mutate(boundary, seed, arm)
        self.fake_summary()
        self.assertEqual(self.run_cli(False), (0, ""))
        report = self.publish.call_args.args[1]
        diagnostic = report["feasibility"]
        self.assertFalse(diagnostic["flags"]["H1_global_floor"])
        self.assertFalse(diagnostic["assay_limited"])
        self.assertEqual(len(diagnostic["per_run"]), 54)
        self.assertTrue(all(r["native_retention_eligible"] and r["paired_native_bytes"]
                            and not r["B_native_below_080"] and not r["common_retention_floor"]
                            for r in diagnostic["per_run"]))
        def weak(row):
            row["metrics"].update(a_acquisition_native=.89, retention_native_delta=-.04,
                                  a_pre_b=.79, retention_delta=-.09)
            row["secondary_matrix"]["fixed"] = .79
        self.mutate(weak)
        self.assertEqual(self.run_cli(False), (0, ""))
        report = self.publish.call_args.args[1]
        diagnostic = report["feasibility"]["per_run"]
        self.assertEqual(len(diagnostic), 54)
        weak_run = next(r for r in diagnostic if r["seed"] == 20 and r["arm"] == "fixed")
        self.assertTrue(weak_run["B_native_below_080"] and weak_run["common_retention_floor"])
        self.assertFalse(weak_run["native_retention_eligible"])
        self.assertEqual(report["no_exclusion"]["included_runs"], 54)
        self.assertEqual(report["no_exclusion"]["excluded_runs"], [])

    def test_frozen_driver_gate_failure_never_reaches_summary(self):
        analyze.driver.read_manifest.side_effect = ValueError("FAKE gate rejected")
        self.blocked()

    def test_existing_output_identical_or_different(self):
        with patch.object(Path, "exists", return_value=True):
            payload = (json.dumps({"FAKE": True}, indent=2, sort_keys=True) + "\n").encode()
            self.fs[str(self.out)] = payload
            # Load the original function kept before patching, without any disk writes.
            self.assertFalse(ORIGINAL_PUBLISH(self.out, {"FAKE": True}))
            with self.assertRaisesRegex(ValueError, "different bytes"):
                ORIGINAL_PUBLISH(self.out, {"FAKE": False})


ORIGINAL_PUBLISH = analyze.publish
ORIGINAL_LOAD_CORE = analyze.load_core

if __name__ == "__main__":
    unittest.main()
