"""ER_fix: standard fixed-capacity reservoir replay baseline for the T2 v4 run (post hoc, addendum 4).

Passport
    What: the pre-registered ER_res arm in t2_mnist.py grows its reservoir by m_replay per
          finished task, so after each growth the first incoming current-task examples enter
          the buffer unconditionally (found by the OpenRouter breaker panel, 2026-11-08).
          This script runs the textbook version (Vitter 1985; Chaudhry et al. 2019): one
          reservoir of fixed capacity m_replay*(T-1) = 200 over the whole stream, replayed at
          B_match's evenly spaced steps with exactly C's replay count in the same seed.
          Same data, model init and current-task / replay RNG streams as t2_mnist.run.
    In:   runs_mnist/curve_seeds30-59.jsonl (C's replay count per bench, tau, seed).
    Out:  runs_mnist/erfix_seeds30-59.jsonl, one row per bench x tau x seed with Ef2_acc.
    Not part of the pre-registered Holm family; reported as a descriptive baseline.
    Rail: local CPU pool, 0 LLM tokens.
"""
import json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

import t2_mnist as m


def run_er_fixed(tasks, seed, k, cfg=m.CFG):
    S, bs, T = cfg["S"], cfg["bs"], m.T
    schedule, _ = m.global_schedules(k, seed, cfg)
    torch.manual_seed(seed)
    model = m.MLP(cfg["hidden"])
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    data_rng = np.random.RandomState(seed * 7 + 1)
    rep_rng = np.random.RandomState(seed * 7 + 2)
    probe_rng = np.random.RandomState(seed * 7 + 5)
    res_rng = np.random.RandomState(seed * 7 + 8)
    cap = cfg["m_replay"] * (T - 1)
    resX, resY, seen, used = [], [], 0, 0
    R = [[0.0] * T for _ in range(T)]

    def step(X, y):
        opt.zero_grad(); F.cross_entropy(model(X), y).backward(); opt.step()

    for t, task in enumerate(tasks):
        Xall, yall = task["train"]
        order = probe_rng.permutation(len(Xall))   # same held-out split as every other arm
        pool = order[cfg["m_probe"]:]
        Xtr, ytr = Xall[pool], yall[pool]
        for s in range(S):
            g = t * S + s
            if t > 0 and g in schedule:
                RX, RY = torch.stack(resX), torch.stack(resY)
                i = rep_rng.randint(0, len(RX), size=bs)
                step(RX[i], RY[i]); used += 1
            else:
                i = data_rng.randint(0, len(Xtr), size=bs)
                step(Xtr[i], ytr[i])
                for j in i:
                    seen += 1
                    if len(resX) < cap:
                        resX.append(Xtr[j]); resY.append(ytr[j])
                    else:
                        r = res_rng.randint(0, seen)
                        if r < cap:
                            resX[r] = Xtr[j]; resY[r] = ytr[j]
        for j in range(T):
            if j == t or t == T - 1:
                X, y = tasks[j]["test"]
                R[t][j] = m.accuracy(model, X, y)
    final = R[T - 1]
    return {"acc": sum(final) / T, "bwt": sum(final[j] - R[j][j] for j in range(T - 1)) / (T - 1), "replay": used}


def job(row):
    torch.set_num_threads(1)
    tasks = m.make_tasks(row["bench"], row["seed"])
    r = run_er_fixed(tasks, row["seed"], row["replay"])
    assert r["replay"] == row["replay"], (r["replay"], row["replay"])
    return {"bench": row["bench"], "tau": row["tau"], "seed": row["seed"], "replay": row["replay"],
            "Ef2_acc": round(r["acc"], 5), "Ef2_bwt": round(r["bwt"], 5)}


def main():
    from multiprocessing import Pool
    src = os.path.join(m.OUT, "curve_seeds30-59.jsonl")
    rows = m.read_jsonl(src)
    m.mnist()
    out = os.path.join(m.OUT, "erfix_seeds30-59.jsonl")
    t0 = time.time()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 14) as p, open(out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(p.imap_unordered(job, rows, chunksize=1)):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if i % 20 == 0:
                print(f"{i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(rows)} in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
