"""T5 form-cost: CPU-only, three paired representation arms, three successive skills.

Method (pilot-dev; freeze this file before fresh-seed inference)
    Records are four categorical fields (0..7). Skills A/B retrieve fields 0/1;
    C computes their maximum, cued by separate task tokens. Splits use the WHOLE tuple;
    eight unordered unequal value pairs (both orders) are a compositional holdout.
    A/B phases use fixed canonical pairs, another fixed permutation, or independent
    per-presentation permutations. All arms see identical records, labels, batches,
    token multisets, lengths and answer positions. Neutral grammar reverses EACH
    key/value pair, preserving vocabulary, and uses an arm-independent field order.
    It is never trained in A/B. H1 is neutral B accuracy after B. H2 is neutral A
    post-B minus pre-B. Phase C trains neutral identically; H3 is its learning AUC
    and final accuracy on the reserved compositions. No replay/recovery is applied.
    DEV-v2 adds one paired A-native post-B probe to anchor preservation in the
    mastered grammar; common H2 remains an operational transfer change and is
    floor-limited if the pre-B common baseline fails. See the dated DEV amendment.
    One contrast per seed: variable - mean(fixed,permuted), two-sided paired t test.
    Variation/augmentation remain confounded; no psychological mechanism claim.

Input/output and consumer
    python t5.py selftest
    python t5.py run --seed 0 --arm fixed --out runs/pilot-dev/seed0_fixed.json
    python t5.py summarize --files runs/pilot-dev/*.json
    The scheduled runs/driver.py calls run. JSON is atomically saved only on full
    completion. Per-checkpoint progress goes to stdout; usage is in result JSON.
    Needs installed torch/numpy/scipy, no downloads. Loss uses a separate label;
    only the ANSWER marker is read. No teacher forcing or gold in inputs.
    Cost ledger counts train/eval tokens, padding (zero), forward/backward calls,
    examples, data and CPU process/wall seconds; FLOPs/energy are unmeasured.
    On failure retain logs, fix instrument before a fresh protocol/run; never infer
    completion from file existence alone. Resource: local CPU, 1 thread by default.
    рельса: Codex subscription for code; deterministic local CPU for training.

"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ARMS = ("fixed", "permuted", "variable")
KEY0, VALUE0, SEP, INSTR, ANSWER, TASK0 = 0, 4, 12, 13, 14, 15
VOCAB, LENGTH = 18, 15
MACHINE = "producer-node"
HELDOUT_PAIRS = {(0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (0, 7), (1, 6), (2, 7)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def code_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def retention(before, after):
    return after - before


def paired_probe_change(before, after):
    assert before["inputs_sha256"] == after["inputs_sha256"], "unpaired retention probe bytes"
    return retention(before["accuracy"], after["accuracy"])


def future_auc(curves, steps):
    c = [r for r in curves if r["phase"] == 2]
    assert c[0]["step"] == 0 and c[-1]["step"] == steps
    return float(np.trapezoid([r["accuracy"] for r in c], [r["step"] for r in c]) / steps)


def target(record, task):
    return record[task] if task < 2 else max(record[0], record[1])


@dataclass(frozen=True)
class Config:
    steps_a: int = 200
    steps_b: int = 200
    steps_c: int = 150
    threads: int = 1
    batch: int = 64
    width: int = 80
    layers: int = 2
    heads: int = 4
    lr: float = .001
    checkpoint: int = 25
    train_n: int = 512
    dev_n: int = 128
    test_n: int = 256
    comp_n: int = 128


def layout(seed):
    # Every block of nine seeds crosses three key rotations and three marker slots.
    rotation = (seed // 3) % 3
    return tuple((k + rotation) % 4 for k in range(4)), seed % 3


def encode(record, task, seed, arm, rng):
    """Atom tokenizer: no BPE; decoder below checks binding, not merely token count."""
    mapping, slot = layout(seed)
    order = list(range(4))
    if arm == "permuted":
        order = [2, 0, 3, 1]
    elif arm in ("variable", "neutral"):
        rng.shuffle(order)
    blocks = []
    for k in order:
        key, value = KEY0 + mapping[k], VALUE0 + record[k]
        blocks.append([value, key, SEP] if arm == "neutral" else [key, value, SEP])
    x = [INSTR] + [t for b in blocks for t in b]
    at = (1, 7, 13)[slot]
    x[at:at] = [TASK0 + task, ANSWER]
    return x, at + 1


def decode(x, seed, neutral=False):
    mapping, _ = layout(seed)
    assert len(x) == LENGTH, "length invariant"
    assert x[0] == INSTR and x.count(ANSWER) == 1, "instruction/answer marker"
    pos = x.index(ANSWER)
    assert TASK0 <= x[pos - 1] < TASK0 + 3, "task marker"
    rest = x[1:pos - 1] + x[pos + 1:]
    result = [None] * 4
    for i in range(0, len(rest), 3):
        a, b, sep = rest[i:i + 3]
        key, value = (b, a) if neutral else (a, b)
        assert sep == SEP and KEY0 <= key < KEY0 + 4, "key grammar"
        assert VALUE0 <= value < VALUE0 + 8, "value grammar"
        k = mapping.index(key - KEY0)
        assert result[k] is None, "duplicate key"
        result[k] = value - VALUE0
    assert None not in result, "missing field"
    return tuple(result)


def validate_forms(record, task, seed, forms):
    reference = forms[0][0]
    for x, pos in forms:
        assert len(x) == len(reference) == LENGTH, "length invariant"
        assert Counter(x) == Counter(reference), "token multiset invariant"
        assert pos == forms[0][1] and x[pos] == ANSWER, "loss position invariant"
        assert decode(x, seed) == tuple(record), "semantic round trip"
        assert x[pos - 1] == TASK0 + task, "task invariant"


def make_data(seed, cfg):
    # Global partitions across ALL tasks prevent cross-task semantic leakage.
    normal, composition = [], []
    for r in itertools.product(range(8), repeat=4):
        (composition if tuple(sorted(r[:2])) in HELDOUT_PAIRS else normal).append(r)
    random.Random(730001 + seed).shuffle(normal)
    pools = dict(train=normal[:2048], dev=normal[2048:2816], test=normal[2816:])
    data = []
    for phase in range(3):
        rng = random.Random(900001 + seed * 101 + phase)
        d = {k: rng.sample(pool, getattr(cfg, k + "_n")) for k, pool in pools.items()}
        d["comp"] = rng.sample(composition, cfg.comp_n)
        data.append(d)
    assert_no_leak(data)
    return data


def assert_no_leak(data):
    grouped = {split: set(r for d in data for r in d[split]) for split in ("train", "dev", "test", "comp")}
    for a, b in itertools.combinations(grouped, 2):
        assert grouped[a].isdisjoint(grouped[b]), "semantic train/test leakage"


def data_manifest(seed, cfg):
    data = make_data(seed, cfg)
    return {"schema": "t5-data/v1", "seed": seed, "layout": layout(seed),
            "data": data, "split_sha256": digest(data), "config": asdict(cfg),
            "rng": {"batches": 200001 + seed, "layout_train": "300001+seed*101+phase",
                    "layout_eval": "400001+seed*101+task*11+split_index",
                    "model": seed}, "Machine": MACHINE}


class Tiny(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, cfg.width)
        self.pos = nn.Embedding(LENGTH, cfg.width)
        layer = nn.TransformerEncoderLayer(cfg.width, cfg.heads, cfg.width * 2,
                                           dropout=0., batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.head = nn.Linear(cfg.width, 8)

    def forward(self, x, at):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1], device="cpu"))
        h = self.encoder(h)
        return self.head(h[torch.arange(len(x)), at])


def batch(records, task, seed, arm, rng):
    pairs = [encode(r, task, seed, arm, rng) for r in records]
    for record, (tokens, position) in zip(records, pairs):
        assert decode(tokens, seed, arm == "neutral") == tuple(record), "live round trip"
        assert position == (2, 8, 14)[seed % 3] and tokens[position] == ANSWER, "live loss position"
    x = torch.tensor([p[0] for p in pairs], dtype=torch.long, device="cpu")
    at = torch.tensor([p[1] for p in pairs], dtype=torch.long, device="cpu")
    y = torch.tensor([target(r, task) for r in records], dtype=torch.long, device="cpu")
    return x, at, y


class Ledger:
    def __init__(self):
        self.rows = {}

    def add(self, category, x, backward=False):
        row = self.rows.setdefault(category, {"tokens": 0, "examples": 0, "forward_calls": 0, "backward_calls": 0})
        row["tokens"] += x.numel()
        row["examples"] += len(x)
        row["forward_calls"] += 1
        row["backward_calls"] += int(backward)

    def total(self):
        return {key: sum(r[key] for r in self.rows.values()) for key in ("tokens", "examples", "forward_calls", "backward_calls")}


class ForwardAudit:
    """Independent instrument attached to actual model calls, not to ledger callers."""
    def __init__(self, model):
        self.tokens = self.calls = 0
        self.handle = model.register_forward_pre_hook(self.observe)

    def observe(self, model, args):
        self.tokens += args[0].numel()
        self.calls += 1

    def verify(self, ledger):
        assert self.tokens == ledger.total()["tokens"], "unmetered forward tokens"
        assert self.calls == ledger.total()["forward_calls"], "unmetered forward calls"


def phase_layout_rng(seed, phase):
    # C must start with identical layout state irrespective of earlier arm's draws.
    return random.Random(300001 + seed * 101 + phase)


@torch.no_grad()
def evaluate(model, records, task, seed, arm, ledger, category):
    model.eval()
    rng = random.Random(400001 + seed * 101 + task * 11 + (0 if category.endswith("dev") else 1))
    correct, loss = 0, 0.
    input_trace = hashlib.sha256()
    for offset in range(0, len(records), 128):
        x, at, y = batch(records[offset:offset + 128], task, seed, arm, rng)
        for tensor in (x, at, y):
            input_trace.update(tensor.numpy().tobytes())
        logits = model(x, at)
        ledger.add(category, x)
        correct += int((logits.argmax(1) == y).sum())
        loss += float(F.cross_entropy(logits, y, reduction="sum"))
    return {"accuracy": correct / len(records), "loss": loss / len(records),
            "inputs_sha256": input_trace.hexdigest()}


def model_sha(model):
    trace = hashlib.sha256()
    for name, value in model.state_dict().items():
        trace.update(name.encode())
        trace.update(value.detach().cpu().numpy().tobytes())
    return trace.hexdigest()


def native_post_probe(model, records, seed, arm, ledger):
    before = model_sha(model)
    rng_before = torch.get_rng_state().clone()
    result = evaluate(model, records, 0, seed, arm, ledger, "retention_native_post_test")
    after = model_sha(model)
    assert before == after, "native probe changed model weights"
    assert torch.equal(rng_before, torch.get_rng_state()), "native probe changed Torch RNG"
    return result, {"weights_before": before, "weights_after": after,
                    "torch_rng_unchanged": True}


def run(seed, arm, cfg):
    initial_sha = code_sha()
    start, process_start = time.perf_counter(), time.process_time()
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    manifest = data_manifest(seed, cfg)
    data = manifest["data"]
    preprocessing = time.perf_counter() - start
    model = Tiny(cfg).to("cpu")
    audit = ForwardAudit(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.)
    ledger, curves = Ledger(), []
    minibatches = random.Random(200001 + seed)
    metrics, matrix, probe_hashes = {}, {}, {}
    semantic_trace = [hashlib.sha256() for _ in range(3)]
    c_input_trace = hashlib.sha256()
    for phase, steps in enumerate((cfg.steps_a, cfg.steps_b, cfg.steps_c)):
        layout_rng = phase_layout_rng(seed, phase)
        form = arm if phase < 2 else "neutral"
        def probe(step):
            # Same neutral bytes at every checkpoint, every arm; no feedback to optimizer.
            ev = evaluate(model, data[phase]["dev"], phase, seed, "neutral", ledger, "probe_dev")
            curves.append({"phase": phase, "step": step, **ev, "cumulative_tokens": ledger.total()["tokens"],
                           "wall_s": time.perf_counter() - start, "cpu_s": time.process_time() - process_start})
            print(json.dumps({"event": "checkpoint", "seed": seed, "arm": arm, **curves[-1]}), flush=True)
        probe(0)
        for step in range(1, steps + 1):
            if time.time() >= float(os.environ.get("T5_RUN_DEADLINE_EPOCH", "inf")):
                raise TimeoutError("pilot cumulative training budget exhausted")
            records = [data[phase]["train"][minibatches.randrange(cfg.train_n)] for _ in range(cfg.batch)]
            semantic_trace[phase].update(json.dumps(records, separators=(",", ":")).encode())
            x, at, y = batch(records, phase, seed, form, layout_rng)
            if phase == 2:
                c_input_trace.update(x.numpy().tobytes())
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x, at), y)
            ledger.add("train_" + "ABC"[phase], x, backward=True)
            loss.backward()
            opt.step()
            if step % cfg.checkpoint == 0 or step == steps:
                probe(step)
        if phase == 0:
            a_common_pre = evaluate(model, data[0]["test"], 0, seed, "neutral", ledger, "retention_pre_test")
            a_native_pre = evaluate(model, data[0]["test"], 0, seed, arm, ledger, "acquisition_native_test")
            metrics["a_pre_b"] = a_common_pre["accuracy"]
            metrics["a_acquisition_native"] = a_native_pre["accuracy"]
            probe_hashes.update(a_common_pre=a_common_pre["inputs_sha256"], a_native_pre=a_native_pre["inputs_sha256"])
        if phase == 1:
            a_common_post = evaluate(model, data[0]["test"], 0, seed, "neutral", ledger, "retention_post_test")
            a_native_post, native_probe_audit = native_post_probe(model, data[0]["test"], seed, arm, ledger)
            metrics["a_post_b"] = a_common_post["accuracy"]
            metrics["a_native_post_b"] = a_native_post["accuracy"]
            metrics["retention_native_delta"] = paired_probe_change(a_native_pre, a_native_post)
            paired_probe_change(a_common_pre, a_common_post)
            probe_hashes.update(a_common_post=a_common_post["inputs_sha256"], a_native_post=a_native_post["inputs_sha256"])
            metrics["h1_b_post"] = evaluate(model, data[1]["test"], 1, seed, "neutral", ledger, "h1_test")["accuracy"]
            metrics["c_comp_pre"] = evaluate(model, data[2]["comp"], 2, seed, "neutral", ledger, "composition_pre_test")["accuracy"]
            for f in (*ARMS, "neutral"):
                matrix[f] = evaluate(model, data[1]["test"], 1, seed, f, ledger, "secondary_matrix")["accuracy"]
    metrics["c_auc"] = future_auc(curves, cfg.steps_c)
    metrics["c_comp"] = evaluate(model, data[2]["comp"], 2, seed, "neutral", ledger, "composition_test")["accuracy"]
    metrics["c_comp_gain"] = metrics["c_comp"] - metrics["c_comp_pre"]
    metrics["retention_delta"] = retention(metrics["a_pre_b"], metrics["a_post_b"])
    audit.verify(ledger)
    assert code_sha() == initial_sha, "source bytes changed during run"
    accounting = {"categories": ledger.rows, "total": ledger.total(), "padding_tokens": 0,
                  "preprocessing_wall_s": preprocessing, "wall_s": time.perf_counter() - start,
                  "cpu_process_s": time.process_time() - process_start,
                  "restoration_steps": 0, "flops": None, "energy_joules": None}
    return {"schema": "t5-result/v1", "status": "complete", "seed": seed, "arm": arm,
            "code_sha256": initial_sha, "config": asdict(cfg), "metrics": metrics,
            "instrument_revision": "native-retention-v2", "probe_hashes": probe_hashes,
            "native_probe_audit": native_probe_audit,
            "curves": curves, "secondary_matrix": matrix, "accounting": accounting,
            "manifest_sha256": os.environ.get("T5_MANIFEST_SHA256", "manual-unregistered"),
            "data_manifest_sha256": digest(manifest), "data_manifest": manifest,
            "training_semantic_sha256": [h.hexdigest() for h in semantic_trace],
            "c_training_inputs_sha256": c_input_trace.hexdigest(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
            "device": "cpu", "Machine": MACHINE,
            "usage": {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "node": MACHINE,
                      "actor": "codex", "event": "experiment.seed_arm", "outcome": "complete"}}


def atomic_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    json.loads(tmp.read_text(encoding="utf-8"))
    os.replace(tmp, path)


def selftest():
    torch.set_num_threads(1)
    cfg = Config(steps_a=1, steps_b=1, steps_c=1)
    assert len({layout(seed) for seed in range(9)}) == 9, "counterbalance cells collapsed"
    assert {encode((0, 1, 2, 3), 0, s, "fixed", random.Random(0))[1] for s in range(9)} == {2, 8, 14}, "answer-position control"
    for seed in range(9):
        for record in [(0, 1, 2, 3), (7, 5, 1, 0), (3, 3, 3, 3)]:
            for task in range(3):
                forms = [encode(record, task, seed, a, random.Random(99)) for a in ARMS]
                validate_forms(record, task, seed, forms)
                neutral, _ = encode(record, task, seed, "neutral", random.Random(99))
                assert decode(neutral, seed, True) == record
                assert all(neutral != f[0] for f in forms), "fourth grammar distinct"
    data = make_data(0, cfg)
    assert_no_leak(data)
    assert all(r[0] != r[1] for d in data for r in d["comp"]), "composition holdout degenerates to retrieval"
    assert {r[0] < r[1] for d in data for r in d["comp"]} == {False, True}, "both comparison orientations required"
    r = data[0]["train"][:8]
    x, at, y = batch(r, 0, 0, "fixed", random.Random(0))
    # Labels live outside model input; changing labels must not change any input.
    y2 = (y + 1) % 8
    x2, at2, _ = batch(r, 0, 0, "fixed", random.Random(0))
    assert torch.equal(x, x2) and torch.equal(at, at2) and not torch.equal(y, y2)
    grads = []
    preds = []
    for alias in ("fixed", "fixed"):
        torch.manual_seed(871)
        m = Tiny(cfg)
        xx, aa, yy = batch(r, 0, 0, alias, random.Random(23))
        pred = m(xx, aa)
        F.cross_entropy(pred, yy).backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters()]))
        preds.append(pred.detach())
    assert torch.equal(grads[0], grads[1]) and torch.equal(preds[0], preds[1]), "placebo gradients"
    ledger = Ledger()
    ledger.add("probe", x)
    before = ledger.total()["tokens"]
    ledger.add("hidden_probe", x)
    assert ledger.total()["tokens"] == before + x.numel(), "hidden probe budget"
    audit_ledger = Ledger()
    audit_model = Tiny(cfg)
    audit = ForwardAudit(audit_model)
    audit_model(x, at)
    audit_ledger.add("metered", x)
    audit.verify(audit_ledger)
    audit_model(x, at)  # actual extra model call, deliberately NOT entered in ledger
    try:
        audit.verify(audit_ledger)
    except AssertionError:
        pass
    else:
        raise AssertionError("hidden forward not detected")
    # Exercise the actual evaluator and retention function with deliberately erased memory.
    class Constant(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, xx, aa):
            return F.one_hot(torch.full((len(xx),), self.value), 8).float() * 20
    fixture = [(3, k, 0, 0) for k in range(8)]
    pre = evaluate(Constant(3), fixture, 0, 0, "neutral", Ledger(), "control_test")["accuracy"]
    post = evaluate(Constant(4), fixture, 0, 0, "neutral", Ledger(), "control_test")["accuracy"]
    assert pre == 1. and post == 0. and retention(pre, post) == -1., "retention detects erased A"
    pre_pair = evaluate(Constant(3), fixture, 0, 0, "fixed", Ledger(), "pre_test")
    post_pair = evaluate(Constant(4), fixture, 0, 0, "fixed", Ledger(), "post_test")
    assert paired_probe_change(pre_pair, post_pair) == -1., "native retention detects erased A"
    broken_pair = dict(post_pair, inputs_sha256="different-probe")
    try:
        paired_probe_change(pre_pair, broken_pair)
    except AssertionError:
        pass
    else:
        raise AssertionError("unpaired retention probes accepted")
    immutable_model = Tiny(cfg)
    state_before = {k: v.clone() for k, v in immutable_model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    _, live_probe_audit = native_post_probe(immutable_model, r, 0, "variable", Ledger())
    assert live_probe_audit["weights_before"] == live_probe_audit["weights_after"]
    assert torch.equal(rng_before, torch.get_rng_state()), "probe advanced model RNG"
    assert all(torch.equal(v, immutable_model.state_dict()[k]) for k, v in state_before.items()), "probe changed weights"
    auc_fixture = [{"phase": 1, "step": 0, "accuracy": 1.},
                   {"phase": 2, "step": 0, "accuracy": 0.},
                   {"phase": 2, "step": 10, "accuracy": 1.}]
    assert future_auc(auc_fixture, 10) == .5, "future metric uses C only"
    # Independent minibatch stream is unaffected by any number of layout shuffles.
    a, b = random.Random(7), random.Random(7)
    form_rng = random.Random(8)
    for _ in range(10):
        encode((0, 1, 2, 3), 0, 0, "variable", form_rng)
    assert [a.randrange(512) for _ in range(20)] == [b.randrange(512) for _ in range(20)]
    c_inputs = []
    for previous_arm in ARMS:
        old_rng = phase_layout_rng(0, 0)
        for _ in range(10):
            encode((0, 1, 2, 3), 0, 0, previous_arm, old_rng)
        c_rng = phase_layout_rng(0, 2)
        c_inputs.append(batch(r, 2, 0, "neutral", c_rng)[0])
    assert all(torch.equal(c_inputs[0], x) for x in c_inputs), "C inputs depend on earlier arm"
    print(json.dumps({"selftest": "PASS", "counterbalance_cells": 9, "parameters": sum(p.numel() for p in Tiny(cfg).parameters()),
                      "code_sha256": code_sha(), "blind_zone": "learnability/runtime/variance require live pilot"}))


def summarize(files):
    from scipy import stats
    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in files]
    assert rows and len({r["code_sha256"] for r in rows}) == 1, "mixed code versions"
    assert len({digest(r["config"]) for r in rows}) == 1, "mixed budgets/configurations"
    assert len({r["manifest_sha256"] for r in rows}) == 1, "mixed run manifests"
    assert rows[0]["manifest_sha256"] != "manual-unregistered", "unregistered results cannot enter paired inference"
    groups = {}
    for row in rows:
        assert row["status"] == "complete" and row["device"] == "cpu"
        seed, arm = row["seed"], row["arm"]
        assert arm not in groups.setdefault(seed, {}), "duplicate seed arm"
        groups[seed][arm] = row
    differences = []
    for seed, arms in sorted(groups.items()):
        assert set(arms) == set(ARMS), "incomplete paired seed"
        assert len({a["code_sha256"] for a in arms.values()}) == 1, "mixed code versions"
        assert len({a["data_manifest_sha256"] for a in arms.values()}) == 1, "unpaired data/config"
        assert len({tuple(a["training_semantic_sha256"]) for a in arms.values()}) == 1, "unpaired actual minibatches"
        assert len({a["c_training_inputs_sha256"] for a in arms.values()}) == 1, "unpaired actual C inputs"
        d = arms["variable"]["metrics"]["h1_b_post"] - sum(arms[a]["metrics"]["h1_b_post"] for a in ARMS[:2]) / 2
        differences.append(d)
    n = len(differences)
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if n > 1 else None
    ci = list(stats.t.interval(.95, n - 1, loc=mean, scale=sd / math.sqrt(n))) if n > 1 and sd else None
    power_n = math.ceil(((1.96 + .84) * sd / .03) ** 2) if sd else None
    print(json.dumps({"n_paired_seeds": n, "differences": differences, "mean": mean, "sd": sd, "ci95": ci,
                      "p_two_sided": float(stats.ttest_1samp(differences, 0).pvalue) if n > 1 and sd else None,
                      "rough_normal_n_for_3pp_80pct": power_n, "power_caveat": "pilot n=3 is unstable; not adequate power evidence",
                      "runs": [{"seed": r["seed"], "arm": r["arm"], "metrics": r["metrics"], "wall_s": r["accounting"]["wall_s"]} for r in rows],
                      "total_wall_s": sum(r["accounting"]["wall_s"] for r in rows)}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--out", required=True)
    for key in ("steps_a", "steps_b", "steps_c", "threads", "batch", "checkpoint"):
        r.add_argument("--" + key.replace("_", "-"), type=int, default=getattr(Config(), key))
    s = sub.add_parser("summarize")
    s.add_argument("--files", nargs="+", required=True)
    args = p.parse_args()
    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "summarize":
        import glob
        summarize([f for pattern in args.files for f in sorted(glob.glob(pattern))])
    else:
        cfg = Config(**{k: getattr(args, k) for k in ("steps_a", "steps_b", "steps_c", "threads", "batch", "checkpoint")})
        assert all(getattr(cfg, k) > 0 for k in ("steps_a", "steps_b", "steps_c", "threads", "batch", "checkpoint"))
        manifest_path = os.environ.get("T5_RUN_MANIFEST")
        if manifest_path:
            raw = Path(manifest_path).read_bytes()
            manifest = json.loads(raw)
            assert hashlib.sha256(raw).hexdigest() == os.environ.get("T5_MANIFEST_SHA256"), "driver manifest hash"
            assert manifest["code_sha256"] == code_sha(), "frozen code mismatch"
            assert manifest["config"] == asdict(cfg), "frozen configuration mismatch"
            assert args.seed in manifest["seeds"] and args.arm in manifest["arms"], "frozen seed/arm mismatch"
        if args.seed not in (0, 1, 2) and (not manifest_path or os.environ.get("T5_CONFIRM_APPROVED") != code_sha()):
            raise SystemExit("fresh seeds locked: frozen exact-SHA review receipt required by driver")
        if Path(args.out).exists():
            raise SystemExit("output exists; validate/resume using driver, never silently overwrite")
        atomic_json(args.out, run(args.seed, args.arm, cfg))


if __name__ == "__main__":
    main()
