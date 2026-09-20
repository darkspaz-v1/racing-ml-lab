"""Regenerate deterministic README screenshots with completed learning curves."""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame

from racing_lab.app import App


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("evolution", "dqn"), default="evolution")
    parser.add_argument("--runs", type=int)
    args = parser.parse_args()
    app = App()
    if args.mode == "dqn":
        app.select_algorithm("dqn")
    target = args.runs or (700 if args.mode == "dqn" else 2)
    while len(app.trainer.history) < target:
        app.trainer.tick()
    for _ in range(16):
        app.trainer.tick()
    app.draw()
    name = "dqn-learning.png" if args.mode == "dqn" else "workbench.png"
    output = Path(__file__).resolve().parent.parent / "assets" / name
    pygame.image.save(app.screen, output)
    pygame.quit()
    print(output)


if __name__ == "__main__":
    main()
