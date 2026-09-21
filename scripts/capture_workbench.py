"""Regenerate deterministic README screenshots with completed learning curves."""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame

from racing_lab.app import App
from racing_lab.simulation import Sensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("evolution", "dqn", "compare", "pitwall", "sandbox", "wiring", "sensors"),
                        default="evolution")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--ticks", type=int, default=180,
                        help="simulation ticks for the comparison screenshot")
    args = parser.parse_args()
    app = App()
    if args.mode == "sensors":
        # A finished "rays only" run first, so the results table has a row to compare.
        app.apply_sensors(Sensors(7, ()))
        while len(app.trainer.history) < 3:
            app.trainer.tick()
        app.apply_sensors(Sensors())
    if args.mode in ("compare", "pitwall"):
        app.start_compare()
        for _ in range(args.ticks):
            app.tick()
    elif args.mode == "dqn":
        app.select_algorithm("dqn")
    if args.mode not in ("compare", "pitwall"):
        target = args.runs or (700 if args.mode == "dqn" else 3 if args.mode == "sensors" else 2)
        while len(app.trainer.history) < target:
            app.trainer.tick()
        for _ in range(16):
            app.trainer.tick()
    if args.mode == "pitwall":
        app.toggle_pit_board()
    elif args.mode == "wiring":
        app.toggle_wiring()
    elif args.mode == "sensors":
        app.toggle_sensor_panel()
        app.sensor_draft = Sensors(11, ("speed", "lane"))
    elif args.mode == "sandbox":
        app.open_sandbox()
        app.sandbox.set_input(3)
        app.sandbox.set_value(0.1)
    app.draw()
    name = {"dqn": "dqn-learning.png", "evolution": "workbench.png",
            "compare": "compare.png", "pitwall": "pit-wall.png",
            "sandbox": "decision-sandbox.png",
            "wiring": "network-wiring.png", "sensors": "sensor-setup.png"}[args.mode]
    output = Path(__file__).resolve().parent.parent / "assets" / name
    pygame.image.save(app.screen, output)
    pygame.quit()
    print(output)


if __name__ == "__main__":
    main()
