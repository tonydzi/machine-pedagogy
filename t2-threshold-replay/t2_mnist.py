"""T2 v4: the T2 threshold-replay test at MNIST scale, 30 fresh seeds, Holm correction.

Passport
    What: re-runs the T2 design (t2.py, v3) on Split-MNIST (class-incremental, 5 tasks of 2
          classes, single 10-way head) and Permuted-MNIST (domain-incremental, 5 pixel
          permutations). Same arms, same fixed gradient-step budget, same held-out probe,
          same separate RNG streams. Adds one standard baseline (ER with a reservoir buffer,
          Chaudhry et al. 2019) at C's replay count, paired Wilcoxon tests, bootstrap 95% CIs
          and Holm correction over all C-vs-control tests (PREREGISTRATION.md, addendum 3).
    Why:  external review 2026-10-27 (PhD-Arxiv-Rejections-Roots-2026-10-27.md, section 3):
          toy scale, n=10, sign test without correction.
    In:   MNIST via sklearn.fetch_openml('mnist_784') (cached in ~/scikit_learn_data).
    Out:  runs_mnist/<tag>.jsonl (one row per bench x tau x seed), stats_mnist.json (all
          numbers used in the paper), redprobe_mnist.json. paper_macros.py turns
          stats_mnist.json into LaTeX macros; nothing in the paper is typed by hand.
    Who calls: a human or agent, `py -3.12 t2_mnist.py --curve --seeds 30-59` then `--stats`.
    Breaks if: OpenML is unreachable and there is no cache.
    Test: _test_t2_mnist.py (red probe + stats arithmetic; it must fail on broken code).
    Rail: local CPU pool, 0 LLM tokens.
"""
import argparse, json, math, os, statistics, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("T2_OUT", os.path.join(HERE, "runs_mnist"))
T = 5
PERM_TAUS = [0.1, 0.15, 0.2, 0.25, 0.3]  # fixed from pilot seeds 0-1 (addendum 3)
TAUS = {"split": [1.0, 1.25, 1.5, 2.0, 2.5], "perm": PERM_TAUS}  # fixed from pilot seeds 0-1 (addendum 3)
CFG = {"S": 500, "N": 5, "bs": 32, "lr": 0.05, "hidden": 256, "m_replay": 50, "m_probe": 20, "val_frac": 0.1}
CONTROLS = [("B_match", "Bm"), ("E_shuf", "Es"), ("E_even", "Ee"), ("F_lin", "Fl"), ("D_rand", "Dr"), ("E_front", "Ef"), ("ER_res", "Er")]
TESTED = ["B_match", "E_shuf", "E_even", "F_lin"]  # the four pre-registered contrasts (Holm family)

_MNIST = None


def mnist():
    global _MNIST
    if _MNIST is None:
        cache = os.path.join(os.path.expanduser("~"), "scikit_learn_data", "mnist_784_t2.npz")
        if not os.path.exists(cache):
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, X=X.astype(np.uint8), y=y.astype(np.int64))
        z = np.load(cache)
        _MNIST = ((z["X"] / 255.0).astype(np.float32), z["y"].astype(np.int64))
    return _MNIST


def make_tasks(bench, seed, cfg=CFG):
    X, y = mnist()
    Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(60000)
    nv = int(cfg["val_frac"] * 60000)
    d = {"train": (Xtr[perm[nv:]], ytr[perm[nv:]]), "val": (Xtr[perm[:nv]], ytr[perm[:nv]]), "test": (Xte, yte)}
    tasks = []
    if bench == "split":
        for t in range(T):
            cls = (2 * t, 2 * t + 1)
            tasks.append({k: (a[np.isin(b, cls)], b[np.isin(b, cls)]) for k, (a, b) in d.items()})
    else:
        prng = np.random.RandomState(1000 + seed)
        for t in range(T):
            p = np.arange(784) if t == 0 else prng.permutation(784)
            tasks.append({k: (a[:, p], b) for k, (a, b) in d.items()})
    return [{k: (torch.from_numpy(np.ascontiguousarray(a)), torch.from_numpy(b)) for k, (a, b) in tk.items()} for tk in tasks]


class MLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 10))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def accuracy(model, X, y):
    return (model(X).argmax(1) == y).float().mean().item()


def run(arm, tasks, seed, cfg=CFG, tau=None, schedule=None, split="test"):
    """One run. Every arm makes T*S gradient steps; a replay step replaces a current-task step.
    ER_res: buffer of the same total size (m_replay per finished task) filled by reservoir
    sampling over every current-task example seen, replayed at the steps in `schedule`."""
    S, N, bs = cfg["S"], cfg["N"], cfg["bs"]
    torch.manual_seed(seed)
    model = MLP(cfg["hidden"])
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    data_rng = np.random.RandomState(seed * 7 + 1)
    rep_rng = np.random.RandomState(seed * 7 + 2)
    buf_rng = np.random.RandomState(seed * 7 + 6)
    probe_rng = np.random.RandomState(seed * 7 + 5)
    res_rng = np.random.RandomState(seed * 7 + 8)
    bufX, bufY, probes = [], [], []
    resX, resY, seen = [], [], 0
    R = [[0.0] * T for _ in range(T)]
    replay_used = diag = 0
    replay_at = []
    credits, recovering = 0, False

    def rise():
        nonlocal diag
        diag += 1
        with torch.no_grad():
            return sum(F.cross_entropy(model(px), py).item() - base for px, py, base in probes) / len(probes)

    def step(X, y):
        opt.zero_grad()
        F.cross_entropy(model(X), y).backward()
        opt.step()

    for t, task in enumerate(tasks):
        Xall, yall = task["train"]
        order = probe_rng.permutation(len(Xall))
        p_idx, pool = order[: cfg["m_probe"]], order[cfg["m_probe"]:]
        Xtr, ytr = Xall[pool], yall[pool]
        cap = cfg["m_replay"] * max(t, 1)  # reservoir capacity = what the per-task buffer holds
        for s in range(S):
            g = t * S + s
            do_replay = False
            if t > 0:
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
                elif schedule is not None:
                    do_replay = g in schedule
            if do_replay:
                if arm == "ER_res":
                    RX, RY = torch.stack(resX), torch.stack(resY)
                else:
                    RX, RY = torch.cat(bufX), torch.cat(bufY)
                i = rep_rng.randint(0, len(RX), size=bs)
                step(RX[i], RY[i])
                replay_used += 1
                replay_at.append(g)
            else:
                i = data_rng.randint(0, len(Xtr), size=bs)
                step(Xtr[i], ytr[i])
                if arm == "ER_res":
                    for j in i:  # reservoir sampling over the stream of current-task examples
                        seen += 1
                        if len(resX) < cap:
                            resX.append(Xtr[j]); resY.append(ytr[j])
                        else:
                            k = res_rng.randint(0, seen)
                            if k < len(resX):
                                resX[k] = Xtr[j]; resY[k] = ytr[j]
        r_idx = buf_rng.permutation(len(Xtr))[: cfg["m_replay"]]
        bufX.append(Xtr[r_idx]); bufY.append(ytr[r_idx])
        with torch.no_grad():
            probes.append((Xall[p_idx], yall[p_idx], F.cross_entropy(model(Xall[p_idx]), yall[p_idx]).item()))
        for j in range(T):  # only the entries ACC and BWT need: the diagonal and the final row
            if j == t or t == T - 1:
                X, y = tasks[j][split]
                R[t][j] = accuracy(model, X, y)
    final = R[T - 1]
    return {"R": R, "acc": sum(final) / T, "bwt": sum(final[j] - R[j][j] for j in range(T - 1)) / (T - 1),
            "replay": replay_used, "replay_at": replay_at, "diag": diag}


def eligible(cfg=CFG):
    return list(range(cfg["S"], T * cfg["S"]))


def global_schedules(k, seed, cfg=CFG):
    el = eligible(cfg)
    even = set() if k == 0 else {el[int((i + 0.5) * len(el) / k)] for i in range(k)}
    rnd = set(np.random.RandomState(seed * 7 + 3).choice(el, size=k, replace=False).tolist())
    return even, rnd


def per_task_schedules(replay_at, seed, cfg=CFG):
    S = cfg["S"]; rng = np.random.RandomState(seed * 7 + 4)
    front, shuf, even = set(), set(), set()
    for t in range(1, T):
        k = sum(1 for g in replay_at if t * S <= g < (t + 1) * S)
        front |= set(range(t * S, t * S + k))
        shuf |= set((t * S + rng.choice(S, size=k, replace=False)).tolist())
        even |= {t * S + int((i + 0.5) * S / k) for i in range(k)}
    return front, shuf, even


def linear_schedule(k, cfg=CFG):
    S = cfg["S"]; w = list(range(1, T)); out = set()
    ks = [int(round(k * x / sum(w))) for x in w]
    ks[-1] += k - sum(ks)
    for t, kt in zip(range(1, T), ks):
        out |= {t * S + int((i + 0.5) * S / kt) for i in range(kt)} if kt else set()
    return out


def curve_job(job):
    bench, tau, sd = job
    torch.set_num_threads(1)
    t0 = time.time()
    tasks = make_tasks(bench, sd)
    c = run("C", tasks, sd, tau=tau)
    even, rnd = global_schedules(c["replay"], sd)
    front, shuf, peven = per_task_schedules(c["replay_at"], sd)
    sched = {"B_match": even, "D_rand": rnd, "E_front": front, "E_shuf": shuf, "E_even": peven,
             "F_lin": linear_schedule(c["replay"]), "ER_res": even}
    row = {"bench": bench, "tau": tau, "seed": sd, "replay": c["replay"], "diag": c["diag"],
           "C_acc": round(c["acc"], 5), "C_bwt": round(c["bwt"], 5)}
    for arm, short in CONTROLS:
        r = run(arm, tasks, sd, schedule=sched[arm])
        assert r["replay"] == c["replay"], (arm, r["replay"], c["replay"])
        row[f"{short}_acc"] = round(r["acc"], 5); row[f"{short}_bwt"] = round(r["bwt"], 5)
    row["sec"] = round(time.time() - t0, 1)
    return row


def anchor_job(job):
    """Reference arms A (no replay) and B (full schedule, 400 replay steps), per seed."""
    bench, sd = job
    torch.set_num_threads(1)
    tasks = make_tasks(bench, sd)
    a, b = run("A", tasks, sd), run("B", tasks, sd)
    return {"bench": bench, "seed": sd, "A_acc": round(a["acc"], 5), "A_bwt": round(a["bwt"], 5),
            "B_acc": round(b["acc"], 5), "B_bwt": round(b["bwt"], 5), "B_replay": b["replay"]}


def red_job(job):
    bench, sd = job
    torch.set_num_threads(1)
    tasks = make_tasks(bench, sd)
    a, b = run("A", tasks, sd), run("B", tasks, sd)
    cn, ca = run("C", tasks, sd, tau=math.inf), run("C", tasks, sd, tau=-math.inf)
    return {"bench": bench, "seed": sd, "C(+inf)==A": cn["R"] == a["R"] and cn["replay"] == 0,
            "C(-inf)==B": ca["R"] == b["R"] and ca["replay"] == b["replay"]}


def pool_map(fn, jobs, workers):
    from multiprocessing import Pool
    with Pool(workers) as p:
        return p.map(fn, jobs, chunksize=1)


# ---------------- statistics (pure functions, tested) ----------------

def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as input."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def boot_ci(diffs, seed=0, B=10000):
    rng = np.random.RandomState(seed)
    d = np.asarray(diffs)
    means = d[rng.randint(0, len(d), size=(B, len(d)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(diffs):
    from scipy.stats import wilcoxon
    d = np.asarray(diffs)
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def stats(rows, anchors):
    out = {"cfg": CFG, "taus": TAUS, "n_seeds": len({r["seed"] for r in rows}),
           "seeds": [min(r["seed"] for r in rows), max(r["seed"] for r in rows)], "points": [], "anchors": {}}
    tests = []
    for bench in ["split", "perm"]:
        an = [a for a in anchors if a["bench"] == bench]
        out["anchors"][bench] = {k: statistics.mean(a[k] for a in an) for k in ["A_acc", "B_acc", "A_bwt", "B_bwt"]}
        out["anchors"][bench]["B_replay"] = an[0]["B_replay"]
        for tau in TAUS[bench]:
            rs = sorted((r for r in rows if r["bench"] == bench and r["tau"] == tau), key=lambda r: r["seed"])
            pt = {"bench": bench, "tau": tau, "n": len(rs), "replay": statistics.mean(r["replay"] for r in rs),
                  "C": statistics.mean(r["C_acc"] for r in rs), "C_sd": statistics.stdev(r["C_acc"] for r in rs)}
            for arm, short in CONTROLS:
                d = [r["C_acc"] - r[f"{short}_acc"] for r in rs]
                pt[arm] = statistics.mean(r[f"{short}_acc"] for r in rs)
                lo, hi = boot_ci(d, seed=int(tau * 100) + (0 if bench == "split" else 7))
                pt[f"d_{arm}"] = statistics.mean(d)
                pt[f"ci_{arm}"] = [lo, hi]
                pt[f"wins_{arm}"] = sum(x > 0 for x in d)
                pt[f"p_{arm}"] = wilcoxon_p(d)
                if arm in TESTED:
                    tests.append((len(out["points"]), arm))
            out["points"].append(pt)
    adj = holm([out["points"][i][f"p_{a}"] for i, a in tests])
    for (i, a), p in zip(tests, adj):
        out["points"][i][f"padj_{a}"] = p
    out["holm_family_size"] = len(tests)
    # pre-registered hypotheses (addendum 3)
    P = lambda b, t: next(p for p in out["points"] if p["bench"] == b and p["tau"] == t)
    h1 = [P("split", t) for t in TAUS["split"][:3]]
    out["H1"] = all(p["d_B_match"] > 0 and p["padj_B_match"] < 0.05 for p in h1)
    out["H2_split"] = all(not (p["padj_E_shuf"] < 0.05 and abs(p["d_E_shuf"]) >= 0.02) for p in out["points"] if p["bench"] == "split")
    out["H2_perm"] = all(not (p["padj_E_shuf"] < 0.05 and abs(p["d_E_shuf"]) >= 0.02) for p in out["points"] if p["bench"] == "perm")
    out["H3"] = all(not (p["d_F_lin"] > 0 and p["padj_F_lin"] < 0.05) for p in out["points"])
    out["H3_flin_sig_better"] = sum(1 for p in out["points"] if p["d_F_lin"] < 0 and p["padj_F_lin"] < 0.05)
    out["n_sig"] = {a: sum(1 for p in out["points"] if p[f"padj_{a}"] < 0.05) for a in TESTED}
    return out


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--anchors", action="store_true")
    ap.add_argument("--red", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--seeds", default="30-59")
    ap.add_argument("--bench", nargs="+", default=["split", "perm"])
    ap.add_argument("--taus", default="", help="comma list; default = the pre-registered grid per bench")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lo, hi = map(int, a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    tag = f"seeds{lo}-{hi}"
    mnist()  # warm the cache once before forking
    if a.red:
        res = pool_map(red_job, [(b, s) for b in a.bench for s in seeds], a.workers)
        json.dump(res, open(os.path.join(HERE, "redprobe_mnist.json"), "w"), indent=1)
        ok = all(r["C(+inf)==A"] and r["C(-inf)==B"] for r in res)
        print("red probe all pass:", ok); sys.exit(0 if ok else 1)
    if a.anchors:
        res = pool_map(anchor_job, [(b, s) for b in a.bench for s in seeds], a.workers)
        with open(os.path.join(OUT, f"anchors_{tag}.jsonl"), "w", encoding="utf-8") as fh:
            for r in res: fh.write(json.dumps(r) + "\n")
        print(f"anchors {len(res)} rows"); return
    if a.curve:
        jobs = [(b, t, s) for b in a.bench for t in ([float(x) for x in a.taus.split(",")] if a.taus else TAUS[b]) for s in seeds]
        t0 = time.time()
        path = os.path.join(OUT, f"curve_{tag}.jsonl")
        from multiprocessing import Pool
        with Pool(a.workers) as p, open(path, "w", encoding="utf-8") as fh:
            for i, r in enumerate(p.imap_unordered(curve_job, jobs, chunksize=1)):
                fh.write(json.dumps(r) + "\n"); fh.flush()
                if i % 10 == 0:
                    print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
        print(f"done {len(jobs)} jobs in {time.time() - t0:.0f}s -> {path}"); return
    if a.stats:
        rows = read_jsonl(os.path.join(OUT, f"curve_{tag}.jsonl"))
        anchors = read_jsonl(os.path.join(OUT, f"anchors_{tag}.jsonl"))
        st = stats(rows, anchors)
        st["torch"] = torch.__version__
        json.dump(st, open(os.path.join(HERE, "stats_mnist.json"), "w"), indent=1)
        print(json.dumps({k: st[k] for k in ["n_seeds", "H1", "H2_split", "H2_perm", "H3", "H3_flin_sig_better", "n_sig"]}))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
