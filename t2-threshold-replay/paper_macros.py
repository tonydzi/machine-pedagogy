"""stats_mnist.json -> LaTeX numbers, table and figure for the JAIR manuscript.

Passport
    What: turns the instrument output (stats_mnist.json from t2_mnist.py --stats, plus the
          v3 digits holdout curve_seeds20-29.csv) into three files in --out:
            numbers.tex     \\tnum{<key>} for every number the text quotes
            table_mnist.tex the benchmark table (threshold vs schedule vs baselines)
            fig_curve.pdf   accuracy against replay steps, both benchmarks
          Nothing numeric in the paper is typed by hand. A key the text asks for but the
          JSON lacks is a LaTeX error (\\num is strict), so stale text cannot compile.
    Who calls: after `t2_mnist.py --stats`; then latexmk in the paper folder.
    Rail: local CPU, 0 LLM tokens.
"""
import argparse, csv, json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = [("B_match", "uniform (global)"), ("E_shuf", "per-task random"), ("E_even", "per-task even (oracle)"),
        ("F_lin", "proportional, no sensor"), ("ER_res", "ER reservoir"), ("D_rand", "random (global)"), ("E_front", "task start")]


def f3(x):
    return f"{x:.3f}".replace("-", "$-$")


def pts(x):  # accuracy difference in percentage points, signed
    return f"{100 * x:+.1f}".replace("-", "$-$")


def pfmt(p):
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def extras(runs=os.path.join(HERE, "runs_mnist")):
    import numpy as np
    sys_path = HERE
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from t2_mnist import boot_ci, read_jsonl
    cur = {(r["bench"], r["tau"], r["seed"]): r for r in read_jsonl(os.path.join(runs, "curve_seeds30-59.jsonl"))}
    erf = {(r["bench"], r["tau"], r["seed"]): r for r in read_jsonl(os.path.join(runs, "erfix_seeds30-59.jsonl"))}
    assert set(erf) == set(cur), "ER_fix rows do not cover the confirmatory run"
    for k in cur:
        assert erf[k]["replay"] == cur[k]["replay"], ("ER_fix replay count differs", k)
    out = {"points": {}, "n": len(erf)}
    for b, tau in sorted({(k[0], k[1]) for k in cur}):
        seeds = sorted(s for (bb, tt, s) in cur if bb == b and tt == tau)
        alloc = [cur[(b, tau, s)]["Ee_acc"] - cur[(b, tau, s)]["Bm_acc"] for s in seeds]
        erfix = [cur[(b, tau, s)]["C_acc"] - erf[(b, tau, s)]["Ef2_acc"] for s in seeds]
        out.setdefault("diag", {})[(b, tau)] = float(np.mean([cur[(b, tau, s)]["diag"] for s in seeds]))
        out["points"][(b, tau)] = {"alloc": float(np.mean(alloc)), "alloc_ci": boot_ci(alloc, seed=11),
                                   "erfix": float(np.mean(erfix)), "erfix_ci": boot_ci(erfix, seed=13)}
    return out


def build(stats, digits_csv, out):
    os.makedirs(out, exist_ok=True)
    N = {}
    N["nseeds"] = str(stats["n_seeds"])
    N["seedlo"], N["seedhi"] = str(stats["seeds"][0]), str(stats["seeds"][1])
    N["holmm"] = str(stats["holm_family_size"])
    N["H1"] = "confirmed" if stats["H1"] else "not confirmed"
    for k in ["H2_split", "H2_perm", "H3"]:  # null-type hypotheses: absence of a significant effect is "not rejected"
        N[k] = "not rejected" if stats[k] else "rejected"
    N["H3flinsig"] = str(stats["H3_flin_sig_better"])
    for a, n in stats["n_sig"].items():
        N[f"nsig-{a}"] = str(n)
    for b, an in stats["anchors"].items():
        for k, v in an.items():
            N[f"{b}-{k}"] = f3(v) if "acc" in k or "bwt" in k else str(v)
    for p in stats["points"]:
        key = f"{p['bench']}-{p['tau']}"
        N[f"{key}-replay"] = f"{p['replay']:.0f}"
        N[f"{key}-C"] = f3(p["C"])
        for a, _ in ARMS:
            N[f"{key}-{a}"] = f3(p[a])
            N[f"{key}-d{a}"] = pts(p[f"d_{a}"])
            N[f"{key}-w{a}"] = str(p[f"wins_{a}"])
            if f"padj_{a}" in p:
                N[f"{key}-p{a}"] = pfmt(p[f"padj_{a}"])
    for b in ["split", "perm"]:
        P = [p for p in stats["points"] if p["bench"] == b]
        for a, _ in ARMS:
            ds = [100 * p[f"d_{a}"] for p in P]
            N[f"{b}-{a}-dmin"], N[f"{b}-{a}-dmax"] = f"{min(ds):+.1f}".replace("-", "$-$"), f"{max(ds):+.1f}".replace("-", "$-$")
        N[f"{b}-replaymin"] = f"{min(p['replay'] for p in P):.0f}"
        N[f"{b}-replaymax"] = f"{max(p['replay'] for p in P):.0f}"
    # derived from the raw per-seed rows: allocation effect (E_even - B_match), timing effect
    # (C - E_even) and the post hoc fixed-capacity reservoir baseline ER_fix (addendum 4)
    X = extras()
    for (b, tau), e in X["points"].items():
        key = f"{b}-{tau}"
        for name in ["alloc", "erfix"]:
            N[f"{key}-{name}"] = pts(e[name])
            N[f"{key}-{name}lo"], N[f"{key}-{name}hi"] = pts(e[name + "_ci"][0]), pts(e[name + "_ci"][1])
    for b in ["split", "perm"]:
        es = [e for (bb, _), e in X["points"].items() if bb == b]
        for name in ["alloc", "erfix"]:
            v = [100 * e[name] for e in es]
            N[f"{b}-{name}-dmin"], N[f"{b}-{name}-dmax"] = f"{min(v):+.1f}".replace("-", "$-$"), f"{max(v):+.1f}".replace("-", "$-$")
        # equivalence check: 95% CI of C - F_lin inside +-2 accuracy points
        P = [p for p in stats["points"] if p["bench"] == b]
        N[f"{b}-equivFlin"] = str(sum(1 for p in P if -0.02 < p["ci_F_lin"][0] and p["ci_F_lin"][1] < 0.02))
    N["erfix-n"] = str(X["n"])
    for (b, tau), v in X["diag"].items():
        N[f"{b}-{tau}-diag"] = f"{v:.0f}"
    for p in stats["points"]:
        p["d_ER_fix"] = X["points"][(p["bench"], p["tau"])]["erfix"]
    # v3 digits holdout (seeds 20-29), for the comparison paragraph
    if digits_csv and os.path.exists(digits_csv):
        rows = list(csv.DictReader(open(digits_csv, encoding="utf-8")))
        for b in ["split", "perm"]:
            d = [float(r["C_acc"]) - float(r["Bm_acc"]) for r in rows if r["bench"] == b]
            N[f"dig-{b}-dBm"] = pts(statistics.mean(d))
    with open(os.path.join(out, "numbers.tex"), "w", encoding="utf-8") as fh:
        fh.write("% generated by paper_macros.py from stats_mnist.json; do not edit\n")
        fh.write("\\makeatletter\n\\newcommand{\\tnum}[1]{\\ifcsname tnum@#1\\endcsname\\csname tnum@#1\\endcsname\\else\\ERRORmissingNumber{#1}\\fi}\n")
        for k, v in sorted(N.items()):
            fh.write(f"\\expandafter\\def\\csname tnum@{k}\\endcsname{{{v}}}\n")
        fh.write("\\makeatother\n")
    # table
    L = []
    for b, name in [("split", "Split-MNIST"), ("perm", "Permuted-MNIST")]:
        for p in [p for p in stats["points"] if p["bench"] == b]:
            cells = [name if p is stats["points"][0] or p["tau"] == stats["taus"][b][0] else "", f"{p['tau']:g}",
                     f"{p['replay']:.0f}", f3(p["C"])]
            for a in ["B_match", "E_shuf", "E_even", "F_lin"]:
                lo, hi = p[f"ci_{a}"]
                star = "$^{*}$" if p[f"padj_{a}"] < 0.05 else ""
                cells.append(f"{pts(p[f'd_{a}'])}{star} \\scriptsize[{100*lo:+.1f}, {100*hi:+.1f}]".replace("[-", "[$-$").replace(", -", ", $-$"))
            cells.append(pts(p["d_ER_res"]))
            cells.append(pts(p["d_ER_fix"]))
            L.append(" & ".join(cells) + " \\\\")
        L.append("\\midrule")
    L = L[:-1]
    with open(os.path.join(out, "table_mnist.tex"), "w", encoding="utf-8") as fh:
        fh.write("% generated by paper_macros.py from stats_mnist.json; do not edit\n" + "\n".join(L) + "\n")
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    style = {"C": ("threshold (C)", "k", "o", "-"), "B_match": ("uniform", "tab:blue", "s", "--"),
             "F_lin": ("proportional, no sensor", "tab:green", "^", "-."), "E_even": ("per-task even (oracle)", "tab:orange", "d", ":"),
             "ER_fix": ("ER reservoir (fixed size)", "tab:gray", "x", "--")}
    for ax, (b, name) in zip(axs, [("split", "Split-MNIST"), ("perm", "Permuted-MNIST")]):
        P = sorted([p for p in stats["points"] if p["bench"] == b], key=lambda p: p["replay"])
        x = [p["replay"] for p in P]
        for a, (lab, c, m, ls) in style.items():
            y = [p["C"] - p["d_ER_fix"] for p in P] if a == "ER_fix" else [p[a] for p in P]
            ax.plot(x, y, color=c, marker=m, ls=ls, lw=1.2, ms=4, label=lab)
        ax.set_title(name, fontsize=9); ax.set_xlabel("replay steps (of 400)", fontsize=8); ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel(f"final mean accuracy ({stats['n_seeds']} seeds)", fontsize=8)
    axs[1].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_curve.pdf"))
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=os.path.join(HERE, "stats_mnist.json"))
    ap.add_argument("--digits", default=os.path.join(HERE, "curve_seeds20-29.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    N = build(json.load(open(a.stats, encoding="utf-8")), a.digits, a.out)
    print(f"wrote {len(N)} numbers, table, figure -> {a.out}")


if __name__ == "__main__":
    main()
