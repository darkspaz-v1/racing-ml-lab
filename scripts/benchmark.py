"""Reproducible, headless learning experiment; no Pygame window required."""

import argparse
import csv
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from racing_lab.learning import DQNTrainer, EvolutionTrainer, Settings
from racing_lab.track import Track, default_track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("evolution", "dqn"), required=True)
    parser.add_argument("--runs", type=int, default=50,
                        help="generations for evolution; episodes for DQN")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--track", type=Path)
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--epsilon-decay", type=float, default=0.997)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    track = Track.load(args.track) if args.track else default_track()
    settings = Settings(seed=args.seed, max_steps=args.max_steps,
                        population=args.population, epsilon_decay=args.epsilon_decay)
    trainer = EvolutionTrainer(track, settings) if args.mode == "evolution" else DQNTrainer(track, settings)
    started = time.perf_counter()
    while len(trainer.history) < args.runs:
        trainer.tick()
        if len(trainer.history) and len(trainer.history) % 50 == 0:
            if len(trainer.history) == 1 or getattr(trainer, "_reported", 0) != len(trainer.history):
                trainer._reported = len(trainer.history)
                completed = sum(row['completion'] > 0 for row in trainer.history)
                evaluated = (sum(row['completion'] > 0 for row in trainer.evaluation_history)
                             if args.mode == "dqn" else None)
                suffix = f", greedy eval laps {evaluated}" if evaluated is not None else ""
                print(f"{len(trainer.history)} runs: best deployable score {trainer.best_score:.1f}, "
                      f"training lap runs {completed}{suffix}", flush=True)
    print(f"Finished in {time.perf_counter() - started:.1f}s. "
          f"Best deployable score {trainer.best_score:.1f}.")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=trainer.history[0].keys())
            writer.writeheader()
            writer.writerows(trainer.history)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
