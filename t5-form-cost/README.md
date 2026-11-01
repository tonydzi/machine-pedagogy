# T5: the cost of a fixed form

Question: if a small network learns the same facts always written in one fixed form,
does it pay for that later, when the same facts arrive in a form it has never seen?
And does a learner that saw the facts in varying forms pay less, at the same training
budget?

## What was tested

Each record has four named fields with values 0 to 7, written as key/value blocks in a
sequence of 15 atomic tokens (vocabulary of 18 IDs, no BPE). Skill A reads field 0,
skill B reads field 1, skill C later computes the maximum of the two. Three arms see
identical records, labels, minibatch order, token multisets, lengths and answer
positions during A and B. Only the form differs:

| Arm | Form during A and B training |
|---|---|
| fixed | one constant block order, [0,1,2,3] |
| fixed-permuted (`permuted`) | another constant block order, [2,0,3,1] |
| variable | a fresh random block order at every presentation |

The exam (primary H1) is accuracy on skill B in a fourth, neutral form that no arm
trained on during A or B: every key/value pair is reversed and the field order does not
depend on the arm. Every arm gets exactly the same budget: 614,400 input positions,
40,960 examples, 595 forward and 550 backward calls per run, counted by a ledger inside
the code. The model is a 2-layer transformer, width 80, 4 heads, 107,448 parameters,
trained on CPU with one thread.

The protocol was written and frozen before any confirmatory data existed
([PREREGISTRATION.md](PREREGISTRATION.md), then the thin revision
[PREREGISTRATION-v2.md](PREREGISTRATION-v2.md); the dated
[DEV-AMENDMENT](DEV-AMENDMENT-2026-10-23.md) and
[ANALYSIS-ADDENDUM](ANALYSIS-ADDENDUM-2026-10-23.md) record what changed after the dev
pilot and why). The pilot used seeds 0 to 2 only ([pilot-analysis.md](pilot-analysis.md),
[pilot-summary.json](pilot-summary.json)) and is excluded from inference. The
confirmatory series used 18 fresh paired seeds, 20 to 37, three arms each: 54 of 54
runs complete, each run once, no exclusions.

The protocol documents and the full result report ([RESULTS.md](RESULTS.md)) are in
Russian with English technical terms; this README is the English summary.

## Primary result

One preregistered test: per seed, the contrast variable minus mean(fixed, permuted) on
neutral-form B accuracy, two-sided paired t test, alpha 0.05.

| Measure | Value |
|---|---:|
| Fixed, neutral-form B accuracy | 12.43% |
| Fixed-permuted, neutral-form B accuracy | 13.80% |
| Variable, neutral-form B accuracy | 35.70% |
| Majority-class baseline on the B test set | 16.25% |
| Paired contrast, variable minus mean of the fixed arms | **+22.58 pp** |
| 95% CI (t, df 17) | [18.96; 26.20] pp |
| Seeds with a positive contrast | 18 of 18 |
| Two-sided p | 2.46e-10 |

Both fixed arms land below the majority baseline of 16.25%: a learner that always saw one
form did not transfer B to the unseen form at all. Only the variable arm rises above the
baseline and above the preregistered floor of 26.25%. So the contrast reads as "variable
form transfers, constant forms fail to transfer", not as a gain on top of working
baselines. An independent recomputation (standard library plus SciPy) agreed on all 18
differences, the mean, CI and p.

Secondary measures (retention of skill A, later learning of skill C, wall time, the full
cost vector) are reported descriptively in [RESULTS.md](RESULTS.md) without extra tests.

## Limitations

- Tiny model, synthetic task, CPU only. Nothing here is shown for large models,
  natural data or real curricula.
- Varying the surface form of training examples is well known as data augmentation.
  This experiment does not claim novelty for that idea and cannot separate form
  variation from augmentation or from familiarity with permutations.
- Retention is not measurable in this assay. The common A probe sat below 0.80 before B
  in all 54 runs, so its post-B change does not measure forgetting. The added native A
  probe is reported per run only, without cross-arm contrasts, and does not establish
  latent forgetting or its absence.
- "Cost" here is engineering accounting of compute (input positions, forward and
  backward calls, examples, CPU seconds), not model welfare or any claim about the
  experience of a network. FLOPs, energy and money were not measured.
- 18 seeds of one architecture and one task give a finite conditional comparison, not a
  power guarantee. No external registration existed before the run; the preregistration
  is frozen by content hashes recorded in the protocol files.

## Reproduce

CPU only, no downloads. Needs Python 3.12, torch, numpy, scipy (the series ran on
torch 2.14.1+cpu, numpy 2.5.3). From this folder:

```
python t5.py selftest
python t5.py summarize --files "runs/confirmatory/native-retention-v2/*.json"
python t5.py run --seed 0 --arm variable --out my-run/seed0_variable.json
```

`summarize` on the 54 stored runs reproduces the primary contrast, CI and p above
(about one minute, most of it importing torch). `run` trains one pilot seed (0, 1 or 2)
in roughly 15 to 25 seconds. Fresh seeds are deliberately locked behind the driver
manifest and review receipt (`runs/driver.py`, `runs/confirmatory-manifest.json`), which
expect the absolute paths of the machine that ran the series; see
[SANITIZATION.md](SANITIZATION.md) for why the strict admission gate in
`runs/analyze.py` verifies the original bytes and not this export. The analyzer's own
unit tests run anywhere:

```
cd runs && python -m unittest test_analysis
```

Data layout: `runs/confirmatory/native-retention-v2/` holds the 54 confirmatory runs,
`runs/pilot-dev/` the 9 dev-v1 runs and `runs/pilot-dev/native-retention-v2/` the 9 dev-v2
runs. `final-summary.json` is the published analysis, `runs/confirmation-independent.json`
the independent recomputation, `runs/confirmation-data-hashes.json` the frozen data hashes.

## Motivation

The question comes from the health-preserving pedagogy of E. N. Dzyatkovskaya (Russian
Academy of Education), which treats the form in which material is presented as a cost the
learner pays, and asks to account for that cost together with what is learned. This
experiment borrows the accounting, not the psychology: it does not transfer human findings
to networks and does not test the mechanism described in that work.

## Integrity of this export

Files were sanitized before publication (internal provenance stamps, local paths and
internal run codes removed; numbers, seeds, statistics and hashes untouched). Hashes
frozen in the preregistration refer to the original bytes. [SANITIZATION.md](SANITIZATION.md)
lists every change class and an `original_sha256 -> exported_sha256` table for each file,
and `tools/sanitize_export.py` is the exact script that produced this folder.
