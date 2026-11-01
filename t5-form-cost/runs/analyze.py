"""Proposed finite admission gate for the T5 confirmatory analysis; not freeze acceptance.

Consumer: this run final analysis after independent review and all 54 CPU runs.
CLI: python -B runs/analyze.py --manifest ABS --out ABS [--check]
Input: exact reviewed confirmatory manifest, frozen data-hash file, seeds20..37
x fixed/permuted/variable, complete JSON plus native-retention-v2 producer audits.
Before statistics: verify driver freeze/review gate, exact file set, frozen embedded
data, matched semantic/C/probe traces, weights/RNG audit, and complete budget.
--check does only read/integrity validation, no core import, inference, output or state writes.
Then core.summarize supplies its unmodified 18-seed contrast/CI/p; this wrapper adds
feasibility/no-exclusion and provenance. It computes no cross-arm native-retention contrast.
Each run retains C dev endpoints and majority/oracle-copy A/B baselines from frozen C records.
Core planning-only rough n is removed; its stale pilot n=3 caveat is relabeled for the fixed18 series.
All observations remain included when assay_limited is true. Output must be outside
the 54-result directory; publication is atomic and never overwrites different bytes.
Producer audit hashes/boolean are checked, not reconstructed model weights/RNG states
(these states are not stored). No training, scheduler, GPU, or external service.
Repair: read the admission error, restore the exact frozen input or complete missing
runs through the reviewed driver; do not remove weak seeds or bypass this entry.
Rail: deterministic standard library; core statistics use its installed SciPy, 0 LLM tokens.
Exit0 validated/check/published; exit1 blocked. Temporary/artifact.proposed until review.
"""
import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import driver

SEEDS = list(range(20, 38))
TOTAL = dict(tokens=614400, examples=40960, forward_calls=595, backward_calls=550)
PROBE = dict(tokens=3840, examples=256, forward_calls=2, backward_calls=0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def probability(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            f"invalid probability: {name}")
    return value


def read_object(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8-sig"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def validate_data(result, frozen, config):
    data = result["data_manifest"]
    require(digest(data) == result["data_manifest_sha256"] == frozen["data_manifest_sha256"],
            "embedded data manifest hash mismatch")
    require(data["seed"] == result["seed"] == frozen["seed"] and data["config"] == config,
            "data seed/config mismatch")
    require(data["layout"] == frozen["layout"], "frozen layout mismatch")
    require(digest(data["data"]) == data["split_sha256"] == frozen["split_sha256"],
            "frozen split hash mismatch")
    require(len(data["data"]) == len(frozen["phases"]) == 3, "three data phases required")
    for phase, expected in zip(data["data"], frozen["phases"]):
        for split in ("train", "dev", "test", "comp"):
            require(len(phase[split]) == expected[split]["n"] == config[split + "_n"],
                    "frozen split count mismatch")
            require(digest(phase[split]) == expected[split]["sha256"], "frozen phase hash mismatch")


def validate_row(row, frozen, m):
    require(row["config"] == m["config"] and row.get("device") == "cpu", "exact config/CPU required")
    require(row.get("instrument_revision") == "native-retention-v2", "native audit revision missing")
    require(row.get("Machine") == m["Machine"], "producer machine mismatch")
    validate_data(row, frozen, m["config"])
    probes = row["probe_hashes"]
    for key in ("a_common_pre", "a_common_post", "a_native_pre", "a_native_post"):
        require(is_hash(probes[key]), f"invalid probe hash: {key}")
    require(probes["a_common_pre"] == probes["a_common_post"] and
            probes["a_native_pre"] == probes["a_native_post"], "unpaired actual A probes")
    audit = row["native_probe_audit"]
    require(is_hash(audit["weights_before"]) and audit["weights_before"] == audit["weights_after"],
            "native probe changed weights or hash missing")
    require(audit["torch_rng_unchanged"] is True, "native probe changed Torch RNG")
    traces = row["training_semantic_sha256"]
    require(len(traces) == 3 and all(is_hash(h) for h in traces) and is_hash(row["c_training_inputs_sha256"]),
            "invalid actual training hashes")
    metrics = row["metrics"]
    for key in ("a_acquisition_native", "a_native_post_b", "a_pre_b", "a_post_b", "h1_b_post", "c_auc",
                "c_comp_pre", "c_comp"):
        probability(metrics[key], key)
    require(type(metrics["c_comp_gain"]) in (int, float) and math.isfinite(metrics["c_comp_gain"])
            and math.isclose(metrics["c_comp_gain"], metrics["c_comp"] - metrics["c_comp_pre"],
                             rel_tol=0, abs_tol=1e-12), "composition gain mismatch")
    curves = row["curves"]
    expected = [(phase, step) for phase, limit in enumerate((m["config"]["steps_a"],
                m["config"]["steps_b"], m["config"]["steps_c"]))
                for step in range(0, limit + 1, m["config"]["checkpoint"])]
    require([(r["phase"], r["step"]) for r in curves] == expected
            and all(type(r["phase"]) is int and type(r["step"]) is int for r in curves),
            "checkpoint phase/step schedule mismatch")
    for checkpoint in curves:
        probability(checkpoint["accuracy"], "checkpoint accuracy")
        require(is_hash(checkpoint["inputs_sha256"]), "invalid actual dev probe hash")
    c = [r for r in curves if r["phase"] == 2]
    auc = sum((a["accuracy"] + b["accuracy"]) * (b["step"] - a["step"]) / 2
              for a, b in zip(c, c[1:])) / m["config"]["steps_c"]
    require(math.isclose(auc, metrics["c_auc"], rel_tol=0, abs_tol=1e-12), "C curve/AUC mismatch")
    probability(row["secondary_matrix"][row["arm"]], "B native")
    require(math.isclose(metrics["retention_native_delta"],
                         metrics["a_native_post_b"] - metrics["a_acquisition_native"], abs_tol=1e-12),
            "native retention delta mismatch")
    require(math.isclose(metrics["retention_delta"], metrics["a_post_b"] - metrics["a_pre_b"], abs_tol=1e-12),
            "common retention delta mismatch")
    ledger = row["accounting"]
    require(ledger["total"] == TOTAL and ledger["categories"]["retention_native_post_test"] == PROBE,
            "total/native-probe budget mismatch")
    for bucket in [ledger["total"], *ledger["categories"].values()]:
        require(set(bucket) == set(TOTAL) and all(type(v) is int and v >= 0 for v in bucket.values()),
                "invalid ledger count")
    require(all(sum(c[k] for c in ledger["categories"].values()) == TOTAL[k] for k in TOTAL),
            "ledger categories do not sum to total")
    require(ledger["padding_tokens"] == 0 and ledger["restoration_steps"] == 0, "unexpected padding/restoration")
    for key in ("wall_s", "cpu_process_s", "preprocessing_wall_s"):
        value = ledger[key]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid runtime")


def validate(manifest):
    m = driver.read_manifest(manifest)
    require(m["phase"] == "confirmatory" and m["seeds"] == SEEDS and m["arms"] == driver.ARMS,
            "exact confirmatory seeds20..37 x3 arms required")
    require(isinstance(m.get("Machine"), str) and m["Machine"] not in ("", "unknown"), "machine missing")
    files = [path for _, _, path in driver.runs(m)]
    observed = list(driver.absolute(m["output_dir"]).iterdir())
    require(len(observed) == 54 and {p.name for p in observed} == {p.name for p in files}
            and all(p.is_file() and not p.is_symlink() for p in observed), "exact set of54 result files required")
    require(driver.all_complete(m), "driver integrity incomplete")
    frozen_path = driver.absolute(m["data_hashes"]["path"])
    frozen, raw_sha = read_object(frozen_path)
    require(raw_sha == m["data_hashes"]["sha256"], "frozen data file raw SHA mismatch")
    require(frozen["schema"] == "t5-frozen-data-hashes/v1" and frozen["seeds"] == SEEDS,
            "frozen seed set mismatch")
    require(frozen["code_sha256"] == m["code_sha256"] and frozen["config"] == m["config"]
            and digest(frozen["config"]) == frozen["config_sha256"], "frozen code/config mismatch")
    require(len(frozen["rows"]) == 18 and {r["seed"] for r in frozen["rows"]} == set(SEEDS),
            "exact18 frozen seed rows required")
    by_seed = {r["seed"]: r for r in frozen["rows"]}
    evidence = {str(frozen_path): raw_sha}
    for path in (m["_manifest_path"], m["t5"], m["gate_receipt"]):
        evidence[str(path)] = driver.sha256(path)
    require(evidence[str(m["t5"])] == m["code_sha256"], "actual core SHA mismatch")
    rows = []
    for seed, arm, path in driver.runs(m):
        ok, reason = driver.valid_result(path, m, seed, arm)
        require(ok, f"invalid result {path.name}: {reason}")
        row, raw_sha = read_object(path)
        require(row["seed"] == seed and row["arm"] == arm and row["code_sha256"] == m["code_sha256"]
                and row["manifest_sha256"] == m["_manifest_sha256"], "result changed during validation")
        validate_row(row, by_seed[seed], m)
        evidence[str(path)] = raw_sha
        rows.append(row)
    for seed in SEEDS:
        group = [r for r in rows if r["seed"] == seed]
        for values in ([tuple(r["training_semantic_sha256"]) for r in group],
                       [r["c_training_inputs_sha256"] for r in group],
                       [r["probe_hashes"]["a_common_pre"] for r in group],
                       [tuple(c["inputs_sha256"] for c in r["curves"]) for r in group]):
            require(len(set(values)) == 1, "unpaired within-seed training/common probe hashes")
    verify_evidence(m, evidence)
    return m, rows, files, evidence


def verify_evidence(m, evidence):
    require(all(driver.sha256(path) == sha for path, sha in evidence.items()), "inputs changed during analysis")
    fresh = driver.read_manifest(m["_manifest_path"])
    require(fresh["_manifest_sha256"] == m["_manifest_sha256"], "manifest changed during analysis")


def c_baselines(records):
    targets = [max(r[0], r[1]) for r in records]
    n = len(records)
    return dict(n=n, majority=max(Counter(targets).values()) / n,
                oracle_copy_A=sum(r[0] == y for r, y in zip(records, targets)) / n,
                oracle_copy_B=sum(r[1] == y for r, y in zip(records, targets)) / n)


def validate_h3_report(diagnostic, config):
    runs = diagnostic["per_run"]
    require(len(runs) == 54 and {(r["seed"], r["arm"]) for r in runs}
            == {(s, a) for s in SEEDS for a in driver.ARMS}, "H3 report requires all54 runs")
    for row in runs:
        for field in ("C_dev_initial", "C_dev_final"):
            probability(row[field], field)
        for split in ("dev", "comp"):
            baseline = row[f"C_{split}_baselines"]
            require(type(baseline["n"]) is int and baseline["n"] == config[split + "_n"],
                    "H3 baseline record count mismatch")
            for field in ("majority", "oracle_copy_A", "oracle_copy_B"):
                probability(baseline[field], f"C {split} {field}")


def feasibility(rows):
    h1, native_b, majority = {}, {}, []
    for arm in driver.ARMS:
        group = [r for r in rows if r["arm"] == arm]
        h1[arm] = sum(r["metrics"]["h1_b_post"] for r in group) / 18
        native_b[arm] = sum(r["secondary_matrix"][arm] for r in group) / 18
    for seed in SEEDS:
        row = next(r for r in rows if r["seed"] == seed)
        records = row["data_manifest"]["data"][1]["test"]
        require(records and all(len(r) == 4 and type(r[1]) is int and 0 <= r[1] < 8 for r in records),
                "invalid actual B labels")
        majority.append(max(Counter(r[1] for r in records).values()) / len(records))
    floor = max(.25, sum(majority) / 18 + .1)
    flags = dict(any_native_A_pre_below_090=any(r["metrics"]["a_acquisition_native"] < .90 for r in rows),
                 any_arm_mean_B_native_below_080=any(v < .80 for v in native_b.values()),
                 H1_global_floor=all(v < floor for v in h1.values()),
                 all_arm_mean_H1_at_least_095=all(v >= .95 for v in h1.values()))
    per_run = []
    for row in rows:
        native_b_score = row["secondary_matrix"][row["arm"]]
        paired = (row["probe_hashes"]["a_native_pre"] == row["probe_hashes"]["a_native_post"])
        c_curve = [c for c in row["curves"] if c["phase"] == 2]
        c_data = row["data_manifest"]["data"][2]
        per_run.append(dict(seed=row["seed"], arm=row["arm"], B_native=native_b_score,
                            B_native_below_080=native_b_score < .8, paired_native_bytes=paired,
                            native_retention_eligible=(row["metrics"]["a_acquisition_native"] >= .9
                                                       and native_b_score >= .8 and paired),
                            common_retention_floor=row["metrics"]["a_pre_b"] < .8,
                            C_dev_initial=c_curve[0]["accuracy"], C_dev_final=c_curve[-1]["accuracy"],
                            C_dev_baselines=c_baselines(c_data["dev"]),
                            C_comp_baselines=c_baselines(c_data["comp"])))
    return dict(assay_limited=any(flags.values()), flags=flags, H1_floor_threshold=floor,
                mean_B_majority=sum(majority) / 18, H1_arm_means=h1, B_native_arm_means=native_b,
                per_run=per_run)


def load_core(path, expected_sha):
    source = Path(path).read_bytes()
    require(hashlib.sha256(source).hexdigest() == expected_sha, "executed core bytes SHA mismatch")
    spec = importlib.util.spec_from_file_location("_t5_frozen_analysis_core", path)
    require(spec is not None and spec.loader is not None, "cannot load frozen core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def publish(path, report):
    payload = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == payload, "output exists with different bytes")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        require(temp.read_bytes() == payload, "output readback mismatch")
        try:
            os.link(temp, path)  # Atomic, no-replace publication on the same filesystem.
        except FileExistsError:
            require(path.read_bytes() == payload, "output created concurrently with different bytes")
        return True
    finally:
        temp.unlink(missing_ok=True)  # Only the temporary file created by this call.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = driver.absolute(args.out)
        m, rows, files, evidence = validate(args.manifest)
        results_dir = driver.absolute(m["output_dir"])
        require(output != results_dir and results_dir not in output.parents,
                "analysis output must be outside the54 immutable result directory")
        diagnostic = feasibility(rows)
        validate_h3_report(diagnostic, m["config"])
        if args.check:
            print(json.dumps(dict(valid=True, runs=54, paired_seeds=18, inference=False,
                                  manifest_sha256=m["_manifest_sha256"])))
            return 0
        core = load_core(m["t5"], m["code_sha256"])
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            core.summarize([str(path) for path in files])
        summary = json.loads(capture.getvalue())
        require(summary["n_paired_seeds"] == 18 and len(summary["differences"]) == 18
                and len(summary["runs"]) == 54, "core summary is not the full18 paired seeds")
        verify_evidence(m, evidence)
        summary.pop("rough_normal_n_for_3pp_80pct", None)
        summary["power_caveat"] = ("Finite 18-seed pilot-scale series; planned power conditional on 3 dev seeds; "
                                   "no post-hoc n adaptation")
        summary.update(schema="t5-final-analysis/v1", status="artifact.proposed", Machine=m["Machine"],
                       feasibility=diagnostic,
                       no_exclusion=dict(included_seeds=SEEDS, included_runs=54, excluded_seeds=[], excluded_runs=[]),
                       provenance=dict(manifest=m["_manifest_path"], manifest_sha256=m["_manifest_sha256"],
                                       code_sha256=m["code_sha256"], input_sha256=evidence,
                                       analyzer_sha256=driver.sha256(__file__)))
        changed = publish(output, summary)
        print(json.dumps(dict(valid=True, published=changed, out=str(output), paired_seeds=18)))
        return 0
    except Exception as exc:
        print(f"T5 analysis BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
