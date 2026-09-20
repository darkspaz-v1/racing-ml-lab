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
