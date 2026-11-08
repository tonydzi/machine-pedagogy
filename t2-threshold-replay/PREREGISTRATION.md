# Pre-registration of the confirmation run (written 2026-10-06 ~06:55 Lisbon, BEFORE running v2)

Why: Codex T3 review found (1) the probe was not held-out (picked from already-trained samples),
(2) B_match mixes per-stage dosage and timing, (3) the headline came from the test curve and a
post-hoc rule (F_lin) with no fresh holdout. v1 outputs are kept in `v1_probe_not_heldout/`.

Fix in v2: the probe set of each task is drawn at task START from a separate RNG and removed from the
training pool, so it is never trained on, neither as a current-task batch nor as replay (the replay buffer is drawn from the remaining pool).

Confirmation: seeds 10-19 (never run before), test split, tau in {0.5, 0.75, 1.0, 1.5, 2.0}, same config.
Hypotheses, decided now:
- H1 (dosage helps): on split, C > B_match at tau in {0.5, 0.75, 1.0} in >= 8/10 paired seeds.
- H2 (timing inside a task does not matter): |C - E_shuf| mean < 0.02 and wins between 3/10 and 7/10.
- H3 (sensor not needed): C does not beat F_lin (wins <= 7/10) at every tau on both benchmarks.
Any outcome is reported as is.

## Addendum 2026-10-06 ~07:15 Lisbon (before running v3)
Codex round 2 found that one RNG chose both buffer exemplars and replay batches, so arms with different
replay timing stored different exemplars. v3 uses a separate buffer RNG. The v2 outputs are kept in `v2_shared_rng/`.
The hypotheses H1-H3 and the seeds 10-19 are unchanged, and v3 is the reported run. Seeds 10-19 have now been seen once
under v2, so v3 is a re-run of a corrected instrument, not a fresh holdout. This is stated in the paper.

## Addendum 2 (2026-10-06 ~07:50 Lisbon, before running seeds 20-29)
Codex round 3: seeds 10-19 were seen under v2, so the v3 run on them is not confirmatory. A fresh holdout,
seeds 20-29 (never run under any version), is now run with code v3 as-is (t2.py, unchanged after this line)
and the same H1-H3 and cutoffs. That run is the confirmatory one; seeds 0-9 and 10-19 become exploratory.
E_even is reported but is an oracle control (it uses C's per-task counts), not a deployable rule.

## Addendum 3 (2026-11-08 15:06 UTC, BEFORE running seeds 30-59): MNIST scale, 30 seeds, Holm (v4)
Why: external review of 2026-10-27 (PhD-Arxiv-Rejections-Roots-2026-10-27.md, section 3): toy scale, n=10, sign test without correction, no standard continual-learning baseline.
Code: t2_mnist.py, sha256 1b1b16d7b0bef35280516a04b5dd936eba9e3e1167f5f4e4c21b388d0eeb42c0 (also in t2_mnist.sha256). Unchanged after this line for the confirmatory run.
Data: MNIST (OpenML mnist_784; first 60,000 = train, 10% of it held out as val per seed; last 10,000 = test). Split-MNIST (class-incremental, 5 tasks of 2 classes, one 10-way head) and Permuted-MNIST (domain-incremental, 5 permutations).
Learner: MLP 784-256-256-10, SGD lr 0.05, batch 32, S=500 steps per task (2,500 total), credit period N=5 (B spends 400 replay steps), 50 stored exemplars and 20 held-out probe images per task.
Arms: C (threshold) and, at exactly C's replay count per seed, B_match, D_rand, E_front, E_shuf, E_even (oracle), F_lin, and ER_res (experience replay with a reservoir buffer of the same size, replay steps evenly spaced; standard baseline, Chaudhry et al. 2019). A and B as anchors.
Tau grid (chosen on exploratory pilot seeds 0-1 so that C's replay count spans the budget; pilot outputs kept in runs_mnist/pilot*): split {1.0, 1.25, 1.5, 2.0, 2.5}; perm {0.1, 0.15, 0.2, 0.25, 0.3}.
Seeds: 30-59 (never run under any version). Test split.
Statistics: per point, paired difference C minus control over 30 seeds; bootstrap 95% CI (10,000 resamples); two-sided Wilcoxon signed-rank p; Holm correction over the full family of 40 tests (4 pre-registered contrasts B_match, E_shuf, E_even, F_lin x 10 points). D_rand, E_front, ER_res are descriptive (no test).
Hypotheses (same direction as H1-H3, adapted to n=30 and Holm):
- H1: on Split-MNIST at the three smallest tau (1.0, 1.25, 1.5), C > B_match with Holm-adjusted p < 0.05 at each.
- H2: within-task timing does not matter: at no point is |C - E_shuf| >= 0.02 with Holm-adjusted p < 0.05 (evaluated per bench).
- H3: the sensor is not needed: at no point does C beat F_lin with Holm-adjusted p < 0.05.
Any outcome is reported as is. The digits study (seeds 0-29, v3) stays in the paper as the original small study.
