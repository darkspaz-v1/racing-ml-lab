# Baseline learning experiment

These results were measured on 2026-09-20, not inferred from the number of lines of code or from a screenshot. Both modes used Foundry Loop, seed 7, and 750 simulation steps per run. Evolution used population 24 and mutation 0.22. DQN used learning rate 0.001, batch size 32, discount 0.97, and ε decay 0.997. The runs used Python 3.13 and NumPy 2.5.3 on Windows.

| Measurement | Evolution | DQN |
|---|---:|---:|
| Training horizon | 30 generations | 800 episodes |
| Best deployable evaluation score | 448.2 | 420.3 |
| First-to-last change | Generation 1: 8% of cars completed a lap; generation 30: 58% | First greedy lap at episode 640; 7 of 80 greedy evaluations completed a lap |
| Training laps | Population completion varies by generation | 112 of 800 exploratory episodes completed a lap |

The **deployable DQN score** is from a separate greedy evaluation, with ε set effectively to zero and no weight update. Training-episode lap counts include random exploratory actions, so they are not evidence that the learned network alone can repeat that drive. This distinction matters: at episode 800, the current DQN network's greedy score had fallen to approximately −2.2, while the saved best checkpoint still scored 420.3. The training curve is unstable; the saved checkpoint prevents a late regression from replacing the best measured driver.

Evolution also has variance: a single generation's best car is not a generalization guarantee. Both scores are **on the training track**. The alternate Harbor Loop is included so you can test transfer. Do not compare these scores as proof that one algorithm is universally stronger; the run budgets and search methods differ.

The later multi-car UI update advances an entire evolution generation concurrently for display. It does not change each policy's physics or score: generations 1–3 still produced best scores 393.5, 398.9, and 421.7 with the original seed and settings. The Compare models tab likewise shows both methods at once but does not equalize how many car steps they consume.

The raw series are [evolution-benchmark.csv](results/evolution-benchmark.csv) and [dqn-eval-benchmark.csv](results/dqn-eval-benchmark.csv). The DQN CSV distinguishes `completion` from `eval_completion`; `eval_score` is filled only every tenth episode. Reproduce the experiments with:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --mode evolution --runs 30 --max-steps 750 --csv data\evolution-repeat.csv
.\.venv\Scripts\python.exe scripts\benchmark.py --mode dqn --runs 800 --max-steps 750 --csv data\dqn-repeat.csv
```

Use the same seed to study the effect of a single setting. With a different seed, learning can happen earlier, later, or fail within the chosen budget. That uncertainty is part of the experiment.

Both benchmarks above use Foundry Loop, which `scripts/benchmark.py` still loads by default, so they reproduce exactly. The app itself now starts on the newer Apex Circuit; pass `--track data/tracks/apex-circuit.json` to benchmark it.

## Sensor comparison on Apex Circuit

Measured 2026-09-20 with evolution only (population 24, mutation 0.22, 750 steps, 30 generations) on Apex Circuit, for five sensor setups and three seeds each (7, 1, 2). The setups change how many inputs the network has, so each is a different network size, not just a different view of the same one.

| Setup (inputs) | First lap at generation (seeds 7 / 1 / 2) | Cars finishing a lap, generations 21–30 (seeds 7 / 1 / 2) | Mean |
|---|---:|---:|---:|
| 7 rays + all 4 extras (12) | 8 / 16 / 13 | 47% / 52% / 52% | 50% |
| 7 rays + speed (8) | 12 / 8 / none | 46% / 38% / 0% | 28% |
| 7 rays only (7) | 12 / 5 / 6 | 55% / 44% / 41% | 47% |
| 0 rays + all 4 extras (5) | none / none / none | 0% / 0% / 0% | 0% |
| 15 rays + all 4 extras (20) | 4 / 8 / 28 | 38% / 47% / 3% | 29% |

What the numbers support:

- **Rays are essential.** With no rays, the car has only route hints and never finished a lap in any seed. Its best score was 27 to 46, against 276 to 281 for every run that finished a lap.
- **The route hints were not needed here.** Seven rays alone (7 inputs) matched the full 12-input setup within seed-to-seed noise: 47% vs 50% late lap rate, and a lap in every seed. Removing hints did not obviously make the task harder on this track.
- **More rays did not help.** Fifteen rays gave the earliest first lap in one seed (generation 4) but the latest in another (generation 28), and a lower average late lap rate. With 20 inputs the network has 400 weights instead of 272, so a fixed mutation setting may search it less efficiently, but three seeds cannot establish that cause.
- **Best score cannot separate the setups.** Every run that finished laps topped out between 276 and 281, because the 750-step limit caps how far any car can go, while runs that never lapped scored 27 to 46 (33 for the one 8-input failure). Lap rate and time-to-first-lap are the informative measures.

What it does not support: a ranking between the 12-input, 8-input and 7-input setups. The 8-input setup failed to finish a lap in one of three seeds, but that is one failure in three, not a demonstrated weakness. These are 30-generation runs on one track with evolution only; DQN was not tested with different sensors, and conclusions may differ on other tracks or with longer training. The raw series are in [`results/sensors/`](results/sensors/).

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --mode evolution --runs 30 --seed 7 --track data\tracks\apex-circuit.json --rays 7 --inputs none --csv data\rays-only.csv
```
