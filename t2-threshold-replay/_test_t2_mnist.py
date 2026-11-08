"""Tests for t2_mnist.py, with a built-in red proof.

Passport
    What: checks the four things the paper's numbers rest on:
          1) red probe: C with tau=+inf is bit-identical to A, with tau=-inf to B;
          2) equal budget: every control arm spends exactly C's replay count;
          3) Holm adjustment matches a hand-computed example;
          4) the hypothesis verdicts in stats() follow the pre-registered rules.
    Red proof: the same checks are run against three deliberately broken copies of the
          module (Holm without the step-down max, C that ignores the credit cap, a control
          schedule that drops one replay step). Each broken copy must FAIL a check,
          otherwise the test itself is fake and this file exits non-zero.
    Run:  py -3.12 _test_t2_mnist.py   (about 1 minute, CPU)
    Rail: local CPU, 0 LLM tokens.
"""
import importlib.util, math, os, sys, tempfile, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "t2_mnist.py")
SMALL = {"S": 40, "N": 5, "bs": 16, "lr": 0.05, "hidden": 32, "m_replay": 10, "m_probe": 10, "val_frac": 0.1}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check_red_probe(m):
    for bench in ["split", "perm"]:
        tasks = m.make_tasks(bench, 7, SMALL)
        a, b = m.run("A", tasks, 7, SMALL), m.run("B", tasks, 7, SMALL)
        cn, ca = m.run("C", tasks, 7, SMALL, tau=math.inf), m.run("C", tasks, 7, SMALL, tau=-math.inf)
        assert cn["R"] == a["R"] and cn["replay"] == 0, f"{bench}: C(+inf) != A"
        assert ca["R"] == b["R"] and ca["replay"] == b["replay"] > 0, f"{bench}: C(-inf) != B"
        assert b["replay"] == (m.T - 1) * SMALL["S"] // SMALL["N"], f"{bench}: B budget {b['replay']}"


def check_equal_budget(m):
    tasks = m.make_tasks("split", 3, SMALL)
    c = m.run("C", tasks, 3, SMALL, tau=0.3)
    assert 0 < c["replay"] <= (m.T - 1) * SMALL["S"] // SMALL["N"], f"C replay {c['replay']} outside (0, B]"
    even, rnd = m.global_schedules(c["replay"], 3, SMALL)
    front, shuf, peven = m.per_task_schedules(c["replay_at"], 3, SMALL)
    for name, s in [("even", even), ("rnd", rnd), ("front", front), ("shuf", shuf), ("peven", peven),
                    ("lin", m.linear_schedule(c["replay"], SMALL))]:
        assert len(s) == c["replay"], f"schedule {name} has {len(s)} steps, C used {c['replay']}"
    r = m.run("ER_res", tasks, 3, SMALL, schedule=even)
    assert r["replay"] == c["replay"], "ER_res replay count differs"


def check_holm(m):
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> step-down max -> 0.06
    want = [0.03, 0.06, 0.06, 0.02]
    got = m.holm(p)
    assert all(abs(g - w) < 1e-12 for g, w in zip(got, want)), f"holm {got} != {want}"


def check_verdicts(m):
    # synthetic rows: C beats B_match by 0.05 on split at every tau, ties elsewhere
    rows, anchors = [], []
    for bench in ["split", "perm"]:
        for sd in range(30):
            anchors.append({"bench": bench, "seed": sd, "A_acc": 0.2, "A_bwt": -1, "B_acc": 0.9, "B_bwt": -0.1, "B_replay": 400})
            for tau in m.TAUS[bench]:
                r = {"bench": bench, "tau": tau, "seed": sd, "replay": 100, "diag": 1, "C_acc": 0.8 + 0.001 * (sd % 5)}
                for _, s in m.CONTROLS:
                    r[f"{s}_acc"] = r["C_acc"] - (0.05 + 0.0001 * sd if (s == "Bm" and bench == "split") else 0.0001 * ((sd % 3) - 1))
                rows.append(r)
    st = m.stats(rows, anchors)
    assert st["H1"] is True, "H1 should be confirmed on synthetic data"
    assert st["H3"] is True, "H3 should hold (C never significantly beats F_lin)"
    assert st["holm_family_size"] == 4 * sum(len(v) for v in m.TAUS.values())


CHECKS = [check_holm, check_verdicts, check_equal_budget, check_red_probe]

MUTANTS = {
    "holm without step-down max": ("running = max(running, min(1.0, (m - rank) * pvals[i]))", "running = min(1.0, (m - rank) * pvals[i])"),
    "C ignores credit cap": ("if credits >= 1 and rise() > tau:", "if rise() > tau:"),
    "per-task schedule drops a step": ("front |= set(range(t * S, t * S + k))", "front |= set(range(t * S, t * S + max(0, k - 1)))"),
}


def run_checks(m):
    failed = []
    for c in CHECKS:
        try:
            c(m)
        except AssertionError as e:
            failed.append(f"{c.__name__}: {e}")
    return failed


def main():
    m = load(SRC, "t2_mnist_under_test")
    failed = run_checks(m)
    for f in failed:
        print("FAIL", f)
    if failed:
        sys.exit(1)
    print(f"PASS {len(CHECKS)} checks on t2_mnist.py")
    src = open(SRC, encoding="utf-8").read()
    fake = []
    for name, (old, new) in MUTANTS.items():
        assert old in src, f"mutant anchor missing: {name}"
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t2_mnist_mutant.py")
            open(p, "w", encoding="utf-8").write(src.replace(old, new, 1))
            mm = load(p, "t2_mnist_mutant_" + str(abs(hash(name))))
            mm._MNIST = m._MNIST
            red = run_checks(mm)
        print(("RED " if red else "GREEN(!) ") + name + (": " + red[0] if red else ""))
        if not red:
            fake.append(name)
    if fake:
        print("FAKE TEST: these broken versions passed:", fake)
        sys.exit(2)
    print("red proof: every broken copy fails")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(); sys.exit(3)
