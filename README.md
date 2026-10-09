# machine-pedagogy

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
