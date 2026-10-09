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
