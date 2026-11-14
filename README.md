# machine-pedagogy

Small, pre-registered experiments that ask whether ideas from healthy teaching of children (rest when the learner is tired, vary the form) also help when training neural networks.

**Who it's for:** machine-learning researchers and reviewers who want to check or challenge the results.
**What you get:** two experiments with code, data and published numbers you can rerun on a normal CPU, including the negative results.

## Quick start
1. Install Python 3 with numpy, scipy, scikit-learn and torch.
2. Run `git clone https://github.com/tonydzi/machine-pedagogy` and `cd machine-pedagogy`.
3. Run `python t5-form-cost/t5.py summarize --files "t5-form-cost/runs/confirmatory/native-retention-v2/*.json"`.

**Done when:** the printed summary shows `"mean": 0.2258...`, the +22.58 points reported for the varying-form arm.

> Want an AI agent to do it for you? Paste this to Claude Code / Codex / Cursor:
> "Read the 'For AI agents' section of https://github.com/tonydzi/machine-pedagogy and set it up for me."

## For AI agents

### Preconditions
- Python 3 (3.12 was used by the authors) on CPU. No GPU, no downloads, no accounts, no keys.
- Packages: numpy, scipy, scikit-learn, torch. There is no requirements file in this repo; install them into a virtual environment and ask the human before installing.
- The long T2 runs (about 65 minutes on 14 CPU workers) are optional; ask the human first.

### Reproduce
```bash
git clone https://github.com/tonydzi/machine-pedagogy
cd machine-pedagogy
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy scipy scikit-learn torch
```
Quick re-derivation of the T5 headline number (read-only, a few seconds; importing torch is the slow part):
```bash
python t5-form-cost/t5.py summarize --files "t5-form-cost/runs/confirmatory/native-retention-v2/*.json"
```
Full T2 reruns (from the repo, see "Details" for what each does):
```bash
python t2-threshold-replay/t2.py --curve --seeds 20-29
python t2-threshold-replay/t2_mnist.py --curve --seeds 30-59
python t2-threshold-replay/_test_t2_mnist.py
```
Warning: `t2.py` writes its output files into its own folder by default (`--out`). Pass `--out <scratch dir>` so the committed results are not overwritten.

### Configure
Nothing to configure.

### Verify
- T5: the summary JSON has `"mean": 0.2258...` (the +22.58 points; 95% CI [18.96; 26.20]). Exit code 0. The summary also lists `"n_paired_seeds": 18`; a different count means a different data folder. It reads committed result files only, so it proves the arithmetic and the files, not that a fresh training run reproduces them.
- Code identity: `sha1sum t2-threshold-replay/t2.py` must equal the contents of `t2-threshold-replay/t2.sha1` (`d81d9acd...` today), and `sha256sum t2-threshold-replay/t2_mnist.py` must equal `t2-threshold-replay/t2_mnist.sha256`. Both matched when this README was written. PowerShell equivalents: `(Get-FileHash -Algorithm SHA1 t2-threshold-replay/t2.py).Hash` and `(Get-FileHash -Algorithm SHA256 t2-threshold-replay/t2_mnist.py).Hash` (upper case; compare ignoring case). If git changed line endings on checkout (autocrlf), the hashes differ for that reason only.
- Test and red proof: `python t2-threshold-replay/_test_t2_mnist.py` must print `PASS 4 checks on t2_mnist.py` and end with `red proof: every broken copy fails`, exit 0 (about 5 minutes on a slow Windows laptop, CPU only; the file itself says about 1 minute on a fast machine). It makes deliberately broken copies of the code and checks that the test catches each, so a green run means the test can actually fail.

### Uninstall / rollback
Delete the cloned folder and the `.venv`. If a rerun overwrote tracked files, restore with `git checkout -- .`.

### Rules for the agent
- Never edit pre-registration files, the superseded `v1_*`/`v2_*` folders, or committed results to make a number match. A failed reproduction is a result: report it.
- Do not change code and keep the old hash files; changed code means the confirmatory claim no longer applies.
- Do not start the 65-minute jobs without the human's yes.

## Details

Experiments that carry health-preserving pedagogy over to how neural networks are trained.
The tradition says teaching has a cost to the learner, and that recovery should start when
the learner's diagnosed state calls for it, not when the timetable does. Machine learning
borrowed the order and dosage parts of pedagogy (curricula, spaced repetition). It did not
borrow the part that protects the learner. These experiments test, one small falsifiable
question at a time, whether that part is worth borrowing.

Every experiment here is pre-registered before its confirmatory run, keeps its superseded
versions in the open, and reports negative results as results.

## T2: recovery on a state signal vs on a schedule

Folder: [`t2-threshold-replay/`](t2-threshold-replay/). Preprint v2 (October 2026):
[doi:10.5281/zenodo.23268660](https://doi.org/10.5281/zenodo.23268660).

Question: at a fixed gradient-step budget in continual learning, does replay triggered by a
loss rise on a held-out probe of past tasks beat replay placed by a schedule?

Result on the pre-registered fresh holdout (seeds 20-29): the threshold arm beats globally
uniform replay (Split-Digits +3.4 to +7.7 points, 8-10 of 10 seeds), but a sensor-free rule
that gives later tasks proportionally more replay matches it at every point and beats it by
6.2 points at the smallest budget. What transfers from the pedagogy is the dosage rule, not
the alarm. Two earlier versions of the instrument were wrong (probe not held out; one RNG
shared between buffer and replay) and were caught by external review; both are kept in
`v1_probe_not_heldout/` and `v2_shared_rng/`.

Reproduce (CPU only, no download; needs numpy, torch, scikit-learn):

```
python t2-threshold-replay/t2.py --all                  # tau tuning, main table, red probe
python t2-threshold-replay/t2.py --curve --seeds 20-29  # confirmatory holdout
```

`t2.sha1` is the hash of the code that produced the confirmatory run.

### T2 v4: MNIST scale, 30 fresh seeds, Holm (November 2026)

Following an external review (toy scale, n=10, uncorrected sign tests), the design was re-run on
Split-MNIST and Permuted-MNIST (MLP 784-256-256-10, 2,500 steps, 400-step replay cap) on fresh
seeds 30-59, with paired Wilcoxon tests, bootstrap 95% CIs and Holm correction over 40
pre-registered tests (addendum 3 of `PREREGISTRATION.md`, code SHA-256 in `t2_mnist.sha256`).
A reservoir experience-replay baseline was added.

Result: on Split-MNIST the threshold arm beats uniform replay at every budget (+5.5 to +12.3
points, all Holm-significant). The sensor-free proportional rule matches it at nine of ten
budgets; at the largest Split budget the trigger wins by +3.6 points (Holm p = 0.001).
On Permuted-MNIST no placement differs. Red probe passes for all 60 runs.

Post hoc (addendum 4): the pre-registered reservoir baseline grew its buffer per task; `er_fixed.py` adds the standard fixed-capacity reservoir (ER_fix), descriptive only.

```
python t2-threshold-replay/t2_mnist.py --curve --seeds 30-59   # ~65 min on 14 CPU workers
python t2-threshold-replay/t2_mnist.py --anchors --seeds 30-59
python t2-threshold-replay/t2_mnist.py --red --seeds 30-59
python t2-threshold-replay/t2_mnist.py --stats --seeds 30-59   # -> stats_mnist.json
python t2-threshold-replay/paper_macros.py --out <paper dir>   # numbers, table, figure
python t2-threshold-replay/_test_t2_mnist.py                   # tests + red proof on 3 mutants
```

## T5: the cost of a fixed form

Folder: [`t5-form-cost/`](t5-form-cost/).

Question: at an equal training budget, does a small network that learned the same records in
one fixed form, another fixed form, or a varying form transfer the skill to a fourth form it
never trained on?

Result on 18 fresh pre-registered paired seeds (20-37): the varying form beats the mean of the
two fixed forms by +22.58 points (95% CI [18.96; 26.20]), positive in 18 of 18 seeds. Both
fixed arms stay below the 16.25% majority baseline, so the contrast is "varying form transfers,
fixed forms fail to transfer". Tiny model, synthetic task, CPU; form variation is known as
augmentation and no novelty is claimed. Files were sanitized before publication;
`t5-form-cost/SANITIZATION.md` maps every original hash to the exported file.

Reproduce (CPU only, no download; needs numpy, torch, scipy):

```
python t5-form-cost/t5.py summarize --files "t5-form-cost/runs/confirmatory/native-retention-v2/*.json"
```

## Planned

T3, persona grain: does a persona trained against a base model's existing dispositions drift
more, or stick harder, than one trained with them, at equal data and compute? Pre-registration
first; nothing here until it exists.

## Who made this

Anton Dziatkovskii (Tony Dzi), Palo Alto AI Research Lab, with E. N. Dzyatkovskaya's published
work on health-preserving pedagogy as the source of the questions. Code reviewed in rounds by an
LLM reviewer (OpenAI Codex), which found both instrument defects above. Corrections and failed
reproductions are welcome as issues.

More where this came from (second brain, agent consensus, persistent memory): github.com/tonydzi

---

<!--ecosystem-map:start-->

## 🧩 One piece of a working system

This repository is one piece lifted out of a live operation: one engineer running a fleet of
machines, built with Claude as implementation collaborator; the machines reach consensus with
each other and wake the human only for money or the irreversible. It was extracted after it
survived production, not written as a demo — and it runs on its own: nothing here phones home
to the rest.

**See how the whole thing fits together → [SYSTEM.md](https://github.com/tonydzi/tonydzi/blob/main/SYSTEM.md)**

<!--ecosystem-map:end-->

## AI contributors

This project is built by a human + AI team, and the git log says so: Claude writes most of
the code, Codex and Grok review it, Gemini feeds the research. Each is credited on a commit
**only if its output changed that commit's content** — no decorative credits. Lab-wide
policy, one source for every repo: [AI-CONTRIBUTORS.md](https://github.com/tonydzi/.github/blob/main/AI-CONTRIBUTORS.md).
