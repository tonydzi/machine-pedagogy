"""T2: threshold-triggered replay vs scheduled replay at a fixed gradient-step budget.

What it does
    Continual learning on 5 tasks built from sklearn's bundled 8x8 digits (no download):
      split : class-incremental, tasks {0,1},{2,3},...,{8,9}, single 10-way head
      perm  : domain-incremental, 5 fixed pixel permutations of all 10 classes
    A small MLP is trained for S gradient steps per task. EVERY arm makes exactly
    T*S gradient steps; a replay step REPLACES a current-task step, never adds one.

Arms
    A        no replay (lower bound)
    B        scheduled replay: one replay step every N steps (full budget)
    C        threshold replay ("recovery phase on a state signal"): every N steps the
             learner earns one replay credit and runs a diagnostic = mean rise of
             cross-entropy on a held-out probe set of past tasks (stored alongside the
             buffer, never trained on) over its value right after that task ended.
             Rise > tau -> recovery phase: replay steps while credits remain and the
             re-diagnosed rise stays > tau. C can never spend more replay than B.
    B_match  evenly spaced replay with exactly C's replay count (same seed)
    D_rand   uniformly random replay timing with exactly C's replay count
    B_match and D_rand separate "how much" from "when": C beating them at equal
    count is the only evidence that state-triggered timing matters.

Honesty rules built in
    tau is tuned on the VALIDATION split with tuning seeds 100-104; final numbers use
    seeds 0-4 on the TEST split. Red probe: tau=+inf must reproduce A bit-for-bit and
    tau=-inf must reproduce B bit-for-bit, otherwise the trigger is not what we claim.

Inputs/outputs
    py -3.12 t2.py --all            # tune + final + red probe for both benchmarks
    py -3.12 t2.py --curve --seeds 0-9    # exploratory cost curve (equal replay counts)
    py -3.12 t2.py --curve --seeds 10-19  # pre-registered confirmation (PREREGISTRATION.md)
    writes tune.csv, results.csv, results.md, redprobe.json, curve_seeds*.csv/.md here.
    Superseded versions are kept: v1_probe_not_heldout/ (probe was trained on) and
    v2_shared_rng/ (one RNG chose both buffer exemplars and replay batches, so arms
    with different replay timing stored different exemplars). Both found by Codex review.
Needs numpy, torch, scikit-learn (CPU is enough, ~1-2 min total).
Rail: local CPU, 0 LLM tokens.
"""
import argparse, csv, json, math, os, statistics, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

HERE = os.path.dirname(os.path.abspath(__file__))
T = 5


def split_data(seed):
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float32)
    rng = np.random.RandomState(seed)
    parts = {"train": [], "val": [], "test": []}
    for c in range(10):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(0.6 * n), int(0.8 * n)
        parts["train"].append(idx[:a]); parts["val"].append(idx[a:b]); parts["test"].append(idx[b:])
    return {k: (X[np.concatenate(v)], y[np.concatenate(v)]) for k, v in parts.items()}


def make_tasks(bench, seed):
    d = split_data(seed)
    tasks = []
    if bench == "split":
        for t in range(T):
            cls = (2 * t, 2 * t + 1)
            task = {}
            for k, (X, y) in d.items():
                m = np.isin(y, cls)
                task[k] = (X[m], y[m])
            tasks.append(task)
    else:
        prng = np.random.RandomState(1000 + seed)
        for t in range(T):
            p = np.arange(64) if t == 0 else prng.permutation(64)
            tasks.append({k: (X[:, p], y) for k, (X, y) in d.items()})
    return [{k: (torch.tensor(X), torch.tensor(y, dtype=torch.long)) for k, (X, y) in tk.items()} for tk in tasks]


class MLP(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 10))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def accuracy(model, X, y):
    return (model(X).argmax(1) == y).float().mean().item()


def run(arm, tasks, seed, cfg, tau=None, schedule=None, split="test"):
    """One training run. Returns dict with acc matrix R[i][j] (after task i, on task j)."""
    S, N, bs = cfg["S"], cfg["N"], cfg["bs"]
    torch.manual_seed(seed)
    model = MLP(cfg["hidden"])
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    data_rng = np.random.RandomState(seed * 7 + 1)   # current-task batches only
    rep_rng = np.random.RandomState(seed * 7 + 2)    # replay batches only
    buf_rng = np.random.RandomState(seed * 7 + 6)    # buffer selection only (v3: same exemplars in every arm)
    bufX, bufY, probes = [], [], []
    R = [[0.0] * T for _ in range(T)]
    replay_used = diag_forwards = 0
    replay_at = []
    credits, recovering = 0, False

    def rise():
        nonlocal diag_forwards
        diag_forwards += 1
        with torch.no_grad():
            return sum(F.cross_entropy(model(px), py).item() - base for px, py, base in probes) / len(probes)

    def step(X, y):
        opt.zero_grad()
        F.cross_entropy(model(X), y).backward()
        opt.step()

    probe_rng = np.random.RandomState(seed * 7 + 5)  # held-out probe selection only
    for t, task in enumerate(tasks):
        Xall, yall = task["train"]
        # probe is set aside BEFORE training on the task and never trained on (v2 fix)
        order = probe_rng.permutation(len(Xall))
        p_idx, pool = order[: cfg["m_probe"]], order[cfg["m_probe"]:]
        Xtr, ytr = Xall[pool], yall[pool]
        for s in range(S):
            g = t * S + s
            do_replay = False
            if bufX:
                if arm == "B":
                    do_replay = (s + 1) % N == 0
                elif arm == "C":
                    if (s + 1) % N == 0:
                        credits += 1
                        recovering = True
                    if recovering:
                        if credits >= 1 and rise() > tau:
                            do_replay = True
                            credits -= 1
                        else:
                            recovering = False
                elif arm in ("B_match", "D_rand", "E_front", "E_shuf", "E_even", "F_lin"):
                    do_replay = g in schedule
            if do_replay:
                RX, RY = torch.cat(bufX), torch.cat(bufY)
                i = rep_rng.randint(0, len(RX), size=bs)
                step(RX[i], RY[i])
                replay_used += 1
                replay_at.append(g)
            else:
                i = data_rng.randint(0, len(Xtr), size=bs)
                step(Xtr[i], ytr[i])
        # end of task: store replay buffer + disjoint probe set, record probe baseline
        r_idx = buf_rng.permutation(len(Xtr))[: cfg["m_replay"]]
        bufX.append(Xtr[r_idx]); bufY.append(ytr[r_idx])
        with torch.no_grad():
            probes.append((Xall[p_idx], yall[p_idx], F.cross_entropy(model(Xall[p_idx]), yall[p_idx]).item()))
        for j in range(T):
            X, y = tasks[j][split]
            R[t][j] = accuracy(model, X, y)
    final = R[T - 1]
    return {
        "R": R,
        "acc": sum(final) / T,
        "bwt": sum(final[j] - R[j][j] for j in range(T - 1)) / (T - 1),
        "replay": replay_used,
        "replay_at": replay_at,
        "diag": diag_forwards,
    }


def eligible_steps(cfg):
    return list(range(cfg["S"], T * cfg["S"]))  # steps after task 0, when a buffer exists


def schedules(k, seed, cfg):
    el = eligible_steps(cfg)
    even = set() if k == 0 else {el[int((i + 0.5) * len(el) / k)] for i in range(k)}
    rnd = set(np.random.RandomState(seed * 7 + 3).choice(el, size=k, replace=False).tolist())
    return even, rnd


def per_task_schedules(replay_at, seed, cfg):
    """Controls that copy C's replay count PER TASK but drop its state-driven timing:
    E_front puts task t's k_t replay steps at the very start of task t (a fixed
    'recovery right after the switch' rule); E_shuf places them at random inside task t;
    E_even spaces them evenly inside task t (added after v3 results: exploratory)."""
    S = cfg["S"]; rng = np.random.RandomState(seed * 7 + 4)
    front, shuf, even = set(), set(), set()
    for t in range(1, T):
        k = sum(1 for g in replay_at if t * S <= g < (t + 1) * S)
        front |= set(range(t * S, t * S + k))
        shuf |= set((t * S + rng.choice(S, size=k, replace=False)).tolist())
        even |= {t * S + int((i + 0.5) * S / k) for i in range(k)}
    return front, shuf, even


def linear_schedule(k, cfg):
    """Sensor-free allocation rule: total k replay steps split across tasks 1..T-1 in
    proportion to the number of past tasks (1:2:3:4), evenly spaced inside each task."""
    S = cfg["S"]; w = list(range(1, T)); out = set()
    ks = [int(round(k * x / sum(w))) for x in w]
    ks[-1] += k - sum(ks)
    for t, kt in zip(range(1, T), ks):
        out |= {t * S + int((i + 0.5) * S / kt) for i in range(kt)} if kt else set()
    return out


def tune(bench, cfg, grid, seeds):
    rows = []
    for tau in grid:
        accs, reps = [], []
        for sd in seeds:
            r = run("C", make_tasks(bench, sd), sd, cfg, tau=tau, split="val")
            accs.append(r["acc"]); reps.append(r["replay"])
        rows.append({"bench": bench, "tau": tau, "val_acc": statistics.mean(accs), "replay": statistics.mean(reps)})
    best = max(rows, key=lambda r: r["val_acc"])
    return best["tau"], rows


def final(bench, cfg, tau, seeds):
    rows = []
    for sd in seeds:
        tasks = make_tasks(bench, sd)
        c = run("C", tasks, sd, cfg, tau=tau)
        even, rnd = schedules(c["replay"], sd, cfg)
        res = {
            "A": run("A", tasks, sd, cfg),
            "B": run("B", tasks, sd, cfg),
            "C": c,
            "B_match": run("B_match", tasks, sd, cfg, schedule=even),
            "D_rand": run("D_rand", tasks, sd, cfg, schedule=rnd),
        }
        for arm, r in res.items():
            rows.append({"bench": bench, "seed": sd, "arm": arm, "tau": tau if arm == "C" else "",
                         "acc": round(r["acc"], 4), "bwt": round(r["bwt"], 4), "replay": r["replay"], "diag": r["diag"]})
    return rows


def red_probe(bench, cfg, seeds):
    out = []
    for sd in seeds:
        tasks = make_tasks(bench, sd)
        a, b = run("A", tasks, sd, cfg), run("B", tasks, sd, cfg)
        c_never = run("C", tasks, sd, cfg, tau=math.inf)
        c_always = run("C", tasks, sd, cfg, tau=-math.inf)
        out.append({"bench": bench, "seed": sd,
                    "C(tau=+inf)==A": c_never["R"] == a["R"] and c_never["replay"] == 0,
                    "C(tau=-inf)==B": c_always["R"] == b["R"] and c_always["replay"] == b["replay"],
                    "A_acc": a["acc"], "C_never_acc": c_never["acc"], "B_acc": b["acc"], "C_always_acc": c_always["acc"],
                    "C_never_replay": c_never["replay"], "B_replay": b["replay"], "C_always_replay": c_always["replay"]})
    return out


def _curve_job(job):
    bench, tau, sd, cfg = job
    torch.set_num_threads(1)
    tasks = make_tasks(bench, sd)
    c = run("C", tasks, sd, cfg, tau=tau)
    even, rnd = schedules(c["replay"], sd, cfg)
    bm = run("B_match", tasks, sd, cfg, schedule=even)
    dr = run("D_rand", tasks, sd, cfg, schedule=rnd)
    front, shuf, even = per_task_schedules(c["replay_at"], sd, cfg)
    ee = run("E_even", tasks, sd, cfg, schedule=even)
    ef = run("E_front", tasks, sd, cfg, schedule=front)
    es = run("E_shuf", tasks, sd, cfg, schedule=shuf)
    fl = run("F_lin", tasks, sd, cfg, schedule=linear_schedule(c["replay"], cfg))
    assert ee["replay"] == fl["replay"] == ef["replay"] == es["replay"] == bm["replay"] == dr["replay"] == c["replay"]
    first = [g - (g // cfg["S"]) * cfg["S"] for g in c["replay_at"]]
    S = cfg["S"]
    share = [sum(1 for g in c["replay_at"] if t * S <= g < (t + 1) * S) for t in range(1, T)]
    return {"C_per_task": "/".join(map(str, share)), "Fl_acc": round(fl["acc"], 4), "Ee_acc": round(ee["acc"], 4), "Ef_acc": round(ef["acc"], 4), "Es_acc": round(es["acc"], 4),
            "C_median_pos_in_task": statistics.median(first) if first else -1,"bench": bench, "tau": tau, "seed": sd, "replay": c["replay"],
            "C_acc": round(c["acc"], 4), "Bm_acc": round(bm["acc"], 4), "Dr_acc": round(dr["acc"], 4),
            "C_bwt": round(c["bwt"], 4), "Bm_bwt": round(bm["bwt"], 4), "Dr_bwt": round(dr["bwt"], 4)}


def curve(benches, cfg, taus, seeds, out):
    """Cost curve, fixed before looking at test: C vs equal-count schedules at several tau."""
    from multiprocessing import Pool
    jobs = [(b, t, s, cfg) for b in benches for t in taus for s in seeds]
    with Pool(min(len(jobs), os.cpu_count() or 4)) as p:
        rows = p.map(_curve_job, jobs)
    tag = f"seeds{seeds[0]}-{seeds[-1]}"
    write_csv(os.path.join(out, f"curve_{tag}.csv"), rows)
    ms = lambda xs: f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}"
    arms = [("B_match", "Bm_acc"), ("D_rand", "Dr_acc"), ("E_front", "Ef_acc"), ("E_shuf", "Es_acc"), ("E_even", "Ee_acc"), ("F_lin", "Fl_acc")]
    lines = ["| bench | tau | replay (of 240) | C replay per task 1-4 (seed 0) | C acc | "
             + " | ".join(f"{a} acc" for a, _ in arms) + " | "
             + " | ".join(f"C - {a} (paired, wins)" for a, _ in arms) + " |",
             "|" + "---|" * (5 + 2 * len(arms))]
    for b in benches:
        for t in taus:
            rs = [r for r in rows if r["bench"] == b and r["tau"] == t]
            cells = [b, str(t), f"{statistics.mean(r['replay'] for r in rs):.0f}",
                     rs[0]["C_per_task"], ms([r["C_acc"] for r in rs])]
            cells += [ms([r[k] for r in rs]) for _, k in arms]
            for _, k in arms:
                d = [r["C_acc"] - r[k] for r in rs]
                cells.append(f"{statistics.mean(d):+.3f} ± {statistics.stdev(d):.3f} ({sum(x > 0 for x in d)}/{len(d)})")
            lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    with open(os.path.join(out, f"curve_{tag}.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# T2 cost curve (test split, seeds {seeds[0]}-{seeds[-1]}, equal replay count per seed)\n\n"
                 f"config: {json.dumps(cfg)}\n\n{table}\n")
    print(table)


def summarize(rows):
    lines = ["| bench | arm | final acc (mean ± sd) | BWT (mean ± sd) | replay steps | diag forwards |",
             "|---|---|---|---|---|---|"]
    for bench in sorted({r["bench"] for r in rows}):
        for arm in ["A", "B", "C", "B_match", "D_rand"]:
            rs = [r for r in rows if r["bench"] == bench and r["arm"] == arm]
            if not rs:
                continue
            f = lambda k: (statistics.mean(r[k] for r in rs), statistics.stdev(r[k] for r in rs))
            (am, asd), (bm, bsd), (rm, rsd), (dm, _) = f("acc"), f("bwt"), f("replay"), f("diag")
            lines.append(f"| {bench} | {arm} | {am:.3f} ± {asd:.3f} | {bm:+.3f} ± {bsd:.3f} | {rm:.0f} ± {rsd:.0f} | {dm:.0f} |")
    return "\n".join(lines)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true", help="tune + final + red probe, both benchmarks")
    ap.add_argument("--bench", nargs="+", default=["split", "perm"])
    ap.add_argument("--S", type=int, default=300, help="gradient steps per task")
    ap.add_argument("--N", type=int, default=5, help="schedule period / credit period")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--curve", action="store_true", help="cost curve: tau in {0.5,0.75,1,1.5,2}, test split")
    ap.add_argument("--seeds", default="0-9", help="curve seeds, e.g. 0-9 (exploratory) or 10-19 (confirmation)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    cfg = {"S": a.S, "N": a.N, "bs": 32, "lr": 0.1, "hidden": 128, "m_replay": 20, "m_probe": 10}
    if a.curve:
        lo, hi = map(int, a.seeds.split("-"))
        curve(a.bench, cfg, [0.5, 0.75, 1.0, 1.5, 2.0], list(range(lo, hi + 1)), a.out); return
    if not a.all:
        ap.print_help(); return
    grid = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    t0 = time.time()
    tune_rows, final_rows, red = [], [], []
    taus = {}
    for bench in a.bench:
        tau, rows = tune(bench, cfg, grid, seeds=[100, 101, 102, 103, 104])
        taus[bench] = tau; tune_rows += rows
        print(f"[{bench}] tuned tau={tau} on val (seeds 100-104)")
        final_rows += final(bench, cfg, tau, seeds=[0, 1, 2, 3, 4])
        red += red_probe(bench, cfg, seeds=[0, 1, 2, 3, 4])
    write_csv(os.path.join(a.out, "tune.csv"), tune_rows)
    write_csv(os.path.join(a.out, "results.csv"), final_rows)
    with open(os.path.join(a.out, "redprobe.json"), "w", encoding="utf-8") as fh:
        json.dump(red, fh, indent=1)
    table = summarize(final_rows)
    md = (f"# T2 results\n\nconfig: {json.dumps(cfg)}; tau grid {grid}; tuned tau {taus}; "
          f"final seeds 0-4 on test; runtime {time.time() - t0:.0f}s; torch {torch.__version__}\n\n{table}\n\n"
          f"red probe all pass: {all(r['C(tau=+inf)==A'] and r['C(tau=-inf)==B'] for r in red)}\n")
    with open(os.path.join(a.out, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
