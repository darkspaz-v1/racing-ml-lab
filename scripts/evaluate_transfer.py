"""Train a learner on one circuit, then evaluate its saved best policy on another.

This is deliberately separate from ``benchmark.py``: a score on the training
circuit says the learner found a policy for that circuit, while this script
tests whether that policy also drives a held-out circuit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racing_lab.learning import DQNTrainer, EvolutionTrainer, Settings
from racing_lab.simulation import Car, EXTRA_INPUTS
from racing_lab.track import Track


def evaluate_policy(network, track: Track, sensors, max_steps: int) -> dict[str, float | int]:
    """Run one deterministic greedy policy without learning or exploration."""
    car = Car(track, max_steps, sensors)
    while not car.done:
        _, outputs = network.forward(car.observation())
        car.step(int(np.argmax(outputs)))
    return {
        "score": car.score(),
        "completion": int(car.laps > 0),
        "progress": min(1.0, car.max_progress / track.total_length),
        "crashed": int(car.crashed),
        "best_lap": min(car.lap_times) if car.lap_times else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("evolution", "dqn"), default="evolution")
    parser.add_argument("--runs", type=int, default=30,
                        help="generations for evolution; episodes for DQN")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 7])
    parser.add_argument("--train-track", type=Path, required=True)
    parser.add_argument("--evaluation-track", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--epsilon-decay", type=float, default=0.997)
    parser.add_argument("--rays", type=int, default=7)
    parser.add_argument("--inputs", default=",".join(EXTRA_INPUTS))
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    train_track = Track.load(args.train_track)
    evaluation_track = Track.load(args.evaluation_track)
    inputs = () if args.inputs.strip().lower() == "none" else tuple(
        name.strip() for name in args.inputs.split(",") if name.strip())
    rows = []
    for seed in args.seeds:
        settings = Settings(seed=seed, max_steps=args.max_steps, population=args.population,
                            epsilon_decay=args.epsilon_decay, rays=args.rays, inputs=inputs)
        trainer = (EvolutionTrainer(train_track, settings) if args.mode == "evolution"
                   else DQNTrainer(train_track, settings))
        while len(trainer.history) < args.runs:
            trainer.tick()
        if trainer.best_network is None:
            raise RuntimeError("Training produced no evaluated policy.")
        source = evaluate_policy(trainer.best_network, train_track, trainer.sensors, args.max_steps)
        held_out = evaluate_policy(trainer.best_network, evaluation_track, trainer.sensors, args.max_steps)
        row = {"seed": seed, "mode": args.mode, "training_track": train_track.name,
               "evaluation_track": evaluation_track.name,
               "training_score": source["score"], "training_completion": source["completion"],
               "held_out_score": held_out["score"], "held_out_completion": held_out["completion"],
               "held_out_progress": held_out["progress"], "held_out_crashed": held_out["crashed"],
               "held_out_best_lap": held_out["best_lap"]}
        rows.append(row)
        print(f"seed {seed}: training {source['score']:.1f}, held-out {held_out['score']:.1f}, "
              f"held-out laps {held_out['completion']}", flush=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
