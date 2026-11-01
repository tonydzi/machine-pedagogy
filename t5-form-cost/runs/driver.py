"""Finite, CPU-only T5 driver; scientific method lives in ../t5.py.

Consumer: Windows task installed by install_tasks.ps1; watch.py restarts this task.
Usage: python driver.py --manifest ABSOLUTE_JSON [--check]
Manifest t5-run-manifest/v1: phase, python, t5, code_sha256, seeds, arms,
config (full t5 Config, including steps_a,steps_b,steps_c,threads:1), output_dir, runtime_dir,
driver_task, watch_task. Optional cpu_sample_interval_seconds (default 5),
run_timeout_seconds (default 3600), gate_receipt (required for confirmatory),
revision ([a-z0-9-]{1,64}, outputs runs/phase/revision; absent keeps legacy path),
pilot_prior_elapsed_seconds (default 0, prior pilot time charged against 1800s).
Pilot seeds are exactly 0,1,2 and all three arms. Invalid prior results stop the
task for review; they are never deleted, moved or overwritten by this driver.
Only complete JSON matching code SHA, raw manifest SHA, seed, arm and budget is skipped.
Three CPU samples must each be <=70% before every run. Above that this task exits 75;
the independent five-minute watch retries. One thread and BelowNormal priority are enforced.
Progress/logs stay in LOCALAPPDATA, separate from scientific JSON in the Vault.
Confirmatory requires hashed preregistration and Codex+Grok ACCEPT receipt;
it is unarmed without that receipt. --check is read-only and checks this gate.
Repair: inspect local events.jsonl/run logs, fix manifest or dependency,
then rerun the driver task; completed, validated outputs are reused.
Rail: deterministic standard library, 0 LLM tokens. Exit 0 complete/check; 1 error;
75 CPU deferred; 76 another driver owns the local lock; 77 finite stop.
Persistent local ledger caps each seed/arm at three attempts and pilot training
at 1800 seconds. CPU waiting is excluded. An interrupted run is conservatively
charged through restart; unknown crash downtime may consume the budget early.
The core also enforces T5_RUN_DEADLINE_EPOCH if this driver itself disappears.
"""
import argparse
import contextlib
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ARMS = ["fixed", "permuted", "variable"]
METRICS = ["h1_b_post", "a_pre_b", "a_post_b", "retention_delta", "c_auc"]
TASK_PREFIX = "T5-form-cost"
CLI_CONFIG = ("steps_a", "steps_b", "steps_c", "threads", "batch", "checkpoint")
STRUCTURAL_DEFAULTS = dict(width=80, layers=2, heads=4, lr=0.001, train_n=512,
                           dev_n=128, test_n=256, comp_n=128)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {path}")
    return data


def absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"absolute path required: {value}")
    return path.resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_manifest(path):
    path = absolute(path)
    m = read_json(path)
    require(m.get("schema") == "t5-run-manifest/v1", "wrong manifest schema")
    require(m.get("phase") in ("pilot-dev", "confirmatory"), "invalid phase")
    for key in ("python", "t5", "output_dir", "runtime_dir"):
        m[key] = str(absolute(m[key]))
    require(Path(m["python"]).is_file(), "python missing")
    require(Path(m["t5"]).name == "t5.py", "exact t5.py required")
    require(sha256(m["t5"]) == m.get("code_sha256"), "t5.py SHA mismatch")
    require(m.get("arms") == ARMS, "exactly fixed/permuted/variable arms required")
    seeds = m.get("seeds")
    require(isinstance(seeds, list) and seeds and all(type(s) is int and s >= 0 for s in seeds)
            and len(set(seeds)) == len(seeds), "invalid seeds")
    if m["phase"] == "pilot-dev":
        require(seeds == [0, 1, 2], "pilot-dev is restricted to seeds 0,1,2")
    else:
        require(not set(seeds) & {0, 1, 2}, "confirmatory cannot reuse pilot seeds")
    config = m.get("config", {})
    require(set(config) == set(CLI_CONFIG) | set(STRUCTURAL_DEFAULTS), "exact full14 config keys required")
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in config.values())
            and type(config["threads"]) is int and config["threads"] == 1,
            "positive budgets and exactly one thread required")
    require(all(type(config[k]) is int for k in CLI_CONFIG), "integer CLI config required")
    require(all(config[k] == v and type(config[k]) is type(v) for k, v in STRUCTURAL_DEFAULTS.items()),
            "non-CLI config must equal frozen core defaults")
    expected = Path(m["t5"]).parent / "runs" / m["phase"]
    if "revision" in m:
        require(type(m["revision"]) is str and re.fullmatch(r"[a-z0-9-]{1,64}", m["revision"]),
                "revision must be safe lowercase ASCII [a-z0-9-]{1,64}")
        expected /= m["revision"]
    require(Path(m["output_dir"]) == expected, "output_dir must match exact phase/revision path")
    prior = m.get("pilot_prior_elapsed_seconds", 0)
    require(type(prior) in (int, float) and math.isfinite(prior) and 0 <= prior < 1800,
            "pilot_prior_elapsed_seconds must be finite and within [0,1800)")
    m["pilot_prior_elapsed_seconds"] = float(prior)
    local = Path(os.environ["LOCALAPPDATA"]) / "T5Agents" / "t5-form-cost"
    require(Path(m["runtime_dir"]) == local.resolve(), "runtime_dir must be the local T5 directory")
    for role in ("driver", "watch"):
        require(m.get(role + "_task") == f"{TASK_PREFIX}-{m['phase']}-{role}", "invalid task name")
    for key, default in (("cpu_sample_interval_seconds", 5), ("run_timeout_seconds", 3600)):
        m[key] = m.get(key, default)
        require(type(m[key]) in (int, float) and math.isfinite(m[key]) and m[key] > 0,
                f"invalid {key}")
    m["_manifest_path"], m["_manifest_sha256"] = str(path), sha256(path)
    validate_gate(m)
    return m


def validate_gate(m):
    if m["phase"] != "confirmatory":
        return
    gate = read_json(absolute(m["gate_receipt"]))
    require(gate.get("schema") == "t5-confirmatory-gate/v1" and gate.get("status") == "ACCEPT",
            "confirmatory gate not ACCEPT")
    for key, expected in (("code_sha256", m["code_sha256"]),
                          ("manifest_sha256", m["_manifest_sha256"])):
        require(gate.get(key) == expected, f"gate {key} mismatch")
    prereg = gate["preregistration"]
    prereg_path = absolute(prereg["path"])
    require(sha256(prereg_path) == prereg["sha256"], "preregistration SHA mismatch")
    require(m["code_sha256"] in prereg_path.read_text(encoding="utf-8"), "code SHA not frozen in prereg")
    reviews = gate.get("reviews", [])
    require(len(reviews) == 2 and {r.get("reviewer") for r in reviews} == {"Codex", "Grok"},
            "Codex and Grok reviews required")
    for review in reviews:
        require(review.get("verdict") == "ACCEPT" and review.get("code_sha256") == m["code_sha256"]
                and review.get("prereg_sha256") == prereg["sha256"], "review targets/verdict mismatch")
        evidence = absolute(review["evidence_path"])
        require(sha256(evidence) == review["evidence_sha256"], "review evidence SHA mismatch")
        text = evidence.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        require(lines and lines[-1] in ("===VERDICT=== ACCEPT", "VERDICT: ACCEPT"),
                "review evidence must end with exact ACCEPT marker")


def runs(m):
    for seed in m["seeds"]:
        for arm in m["arms"]:
            yield seed, arm, Path(m["output_dir"]) / f"seed{seed}_{arm}.json"


def valid_result(path, m, seed, arm):
    try:
        r = read_json(path)
        expected = {"schema": "t5-result/v1", "status": "complete", "seed": seed, "arm": arm,
                    "code_sha256": m["code_sha256"], "manifest_sha256": m["_manifest_sha256"]}
        require(all(r.get(k) == v and type(r.get(k)) is type(v) for k, v in expected.items()),
                "result identity/SHA/status mismatch")
        require(isinstance(r.get("config"), dict), "result config must be an object")
        require(all(r.get("config", {}).get(k) == v and type(r["config"][k]) is type(v)
                    for k, v in m["config"].items()), "budget mismatch")
        for key in METRICS:
            value = r["metrics"][key]
            require(type(value) in (int, float) and math.isfinite(value), f"invalid metric {key}")
        require(isinstance(r.get("curves"), list) and r["curves"], "curves missing")
        require(isinstance(r.get("accounting"), dict) and r["accounting"], "accounting missing")
        return True, "valid"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, str(exc)


def all_complete(m):
    return all(valid_result(path, m, seed, arm)[0] for seed, arm, path in runs(m))


def event(m, name, outcome, **extra):
    runtime = Path(m["runtime_dir"])
    runtime.mkdir(parents=True, exist_ok=True)
    record = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), node="producer-node",
                  actor="Codex/robot", event=name, outcome=outcome, **extra)
    line = json.dumps(record, ensure_ascii=False)
    with (runtime / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def state(m, **extra):
    path = Path(m["runtime_dir"]) / "status.json"
    value = dict(updated=time.time(), manifest_sha256=m["_manifest_sha256"],
                 driver_identity=process_identity(os.getpid()), **extra)
    atomic_json(path, value)


def atomic_json(path, value):
    temp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    require(read_json(temp) == value, "local status readback failed")
    os.replace(temp, path)


def budget_path(m):
    return Path(m["runtime_dir"]) / f"budget-{m['_manifest_sha256']}.json"


def budget(m):
    path = budget_path(m)
    return read_json(path) if path.exists() else dict(elapsed=0.0, attempts={}, inflight=None)


def pilot_limit(m):
    return 1800 - m.get("pilot_prior_elapsed_seconds", 0)


def stop_reason(m):
    value = budget(m)
    elapsed = value["elapsed"] + (max(0, time.time() - value["inflight"]) if value["inflight"] else 0)
    if m["phase"] == "pilot-dev" and elapsed >= pilot_limit(m):
        return "pilot_budget_exhausted"
    for seed, arm, path in runs(m):
        if not valid_result(path, m, seed, arm)[0]:
            if path.exists():
                return "invalid_output"
            if value["attempts"].get(f"{seed}_{arm}", 0) >= 3:
                return "attempts_exhausted"
    return None


def lower_priority():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    require(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000), "cannot lower driver priority")


def process_identity(pid, handle=None):
    if os.name != "nt":
        return dict(pid=pid, creation_filetime=None)
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    values = [wintypes.FILETIME() for _ in range(4)]
    require(kernel.GetProcessTimes(handle if handle is not None else kernel.GetCurrentProcess(),
                                   *(ctypes.byref(v) for v in values)), "process birth unavailable")
    return dict(pid=pid, creation_filetime=values[0].dwLowDateTime + (values[0].dwHighDateTime << 32))


@contextlib.contextmanager
def driver_lock(m):
    path = Path(m["runtime_dir"]) / "driver.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BlockingIOError("local driver lock held") from exc
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def cpu_times():
    from ctypes import wintypes
    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    require(ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel),
                                                ctypes.byref(user)), "CPU counter unavailable")
    number = lambda value: value.dwLowDateTime + (value.dwHighDateTime << 32)
    return number(idle), number(kernel) + number(user)


def cpu_samples(interval):
    previous, values = cpu_times(), []
    for _ in range(3):
        time.sleep(interval)
        current = cpu_times()
        idle, total = current[0] - previous[0], current[1] - previous[1]
        require(total > 0 and 0 <= idle <= total, "invalid CPU counter delta")
        values.append(100 * (1 - idle / total))
        previous = current
    return values


def command(m, seed, arm, output):
    args = [m["python"], "-B", "-u", m["t5"], "run", "--seed", str(seed), "--arm", arm, "--out", str(output)]
    for key in CLI_CONFIG:
        args.extend(["--" + key.replace("_", "-"), str(m["config"][key])])
    return args


def execute(m):
    event(m, "driver", "start", phase=m["phase"])
    try:
        with driver_lock(m):
            if os.name == "nt":
                lower_priority()
            ledger = budget(m)
            if ledger["inflight"]:
                ledger["elapsed"] += max(0, time.time() - ledger["inflight"])
                ledger["inflight"] = None
                atomic_json(budget_path(m), ledger)
            for seed, arm, output in runs(m):
                valid, reason = valid_result(output, m, seed, arm)
                if valid:
                    event(m, "result", "skip_valid", seed=seed, arm=arm)
                    continue
                require(sha256(m["t5"]) == m["code_sha256"], "t5 changed during driver")
                validate_gate(m)
                stopped = stop_reason(m)
                if stopped:
                    state(m, phase=stopped, seed=seed, arm=arm)
                    event(m, "driver", stopped)
                    return 77
                state(m, phase="cpu_gate", seed=seed, arm=arm)
                samples = cpu_samples(m["cpu_sample_interval_seconds"])
                event(m, "cpu_gate", "pass" if max(samples) <= 70 else "defer", samples=samples)
                if max(samples) > 70:
                    return 75
                output.parent.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                log = Path(m["runtime_dir"]) / f"seed{seed}_{arm}-{stamp}-{os.getpid()}.log"
                env = dict(os.environ, T5_MANIFEST_SHA256=m["_manifest_sha256"],
                           T5_RUN_MANIFEST=m["_manifest_path"], CUDA_VISIBLE_DEVICES="",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                for key in ("T5_CONFIRM_APPROVED", "T5_RUN_DEADLINE_EPOCH"):
                    env.pop(key, None)
                if m["phase"] == "confirmatory":
                    env["T5_CONFIRM_APPROVED"] = m["code_sha256"]
                timeout = m["run_timeout_seconds"]
                if m["phase"] == "pilot-dev":
                    timeout = min(timeout, pilot_limit(m) - ledger["elapsed"])
                    env["T5_RUN_DEADLINE_EPOCH"] = str(time.time() + timeout)
                event(m, "run", "start", seed=seed, arm=arm, previous=reason, log=str(log))
                ledger["attempts"][f"{seed}_{arm}"] = ledger["attempts"].get(f"{seed}_{arm}", 0) + 1
                ledger["inflight"] = time.time()
                atomic_json(budget_path(m), ledger)
                started = time.monotonic()
                try:
                    with log.open("wb") as stream:
                        result = subprocess.Popen(command(m, seed, arm, output), env=env, stdout=stream,
                                                  stderr=subprocess.STDOUT,
                                                  creationflags=0x08004000 if os.name == "nt" else 0)
                        try:
                            state(m, phase="running", seed=seed, arm=arm,
                                  worker_identity=process_identity(result.pid, getattr(result, "_handle", None)))
                            result.wait(timeout=timeout)
                        except BaseException as exc:
                            result.kill()
                            result.wait()
                            if isinstance(exc, subprocess.TimeoutExpired):
                                event(m, "run", "timeout", seed=seed, arm=arm, timeout_seconds=timeout)
                            raise
                finally:
                    ledger["elapsed"] += time.monotonic() - started
                    ledger["inflight"] = None
                    atomic_json(budget_path(m), ledger)
                valid, reason = valid_result(output, m, seed, arm)
                event(m, "run", "valid" if valid and result.returncode == 0 else "failed",
                      seed=seed, arm=arm, returncode=result.returncode, validation=reason)
                require(result.returncode == 0 and valid, "run failed or result invalid; watch will retry")
            require(all_complete(m), "final readback incomplete")
            state(m, phase="complete")
            event(m, "driver", "complete", validated=len(m["seeds"]) * len(m["arms"]))
            return 0
    except BlockingIOError:
        event(m, "driver", "already_locked")
        return 76


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        m = read_manifest(args.manifest)
        if args.check:
            print(json.dumps(dict(valid=True, complete=all_complete(m), manifest_sha256=m["_manifest_sha256"])))
            return 0
        return execute(m)
    except Exception as exc:
        print(f"T5 driver ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
