# Held-out track evaluation

A training score alone cannot show that a racing policy transfers beyond the
circuit it repeatedly saw. `scripts/evaluate_transfer.py` trains a learner on
one track, freezes its best evaluated policy, and drives that same policy on a
different bundled track with no weight updates or exploration.

For an evolution experiment from Apex Circuit to Harbor Loop:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_transfer.py `
  --mode evolution --runs 30 --seeds 1 2 7 `
  --train-track data\tracks\apex-circuit.json `
  --evaluation-track data\tracks\harbor-loop.json `
  --csv docs\results\apex-to-harbor-evolution.csv
```

The CSV records one deterministic policy evaluation per seed. Report the
number of held-out lap completions and the individual scores; do not average
away failures. This is a transfer check, not a benchmark against other
projects or a claim of real-world autonomous-driving performance.

The included learners were designed to make learning behavior visible, not to
maximize generalization. A low held-out score is a useful result: it shows the
policy has specialized to its training circuit.
