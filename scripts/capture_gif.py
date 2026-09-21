"""Render a short animated GIF of evolution training (headless, needs Pillow).

    python scripts/capture_gif.py            # writes assets/evolution.gif

The window is never shown: SDL's dummy video driver is used. The first
``--warmup`` generations are fast-forwarded, then the start of each of the next
``--generations`` generations is recorded (``--frames`` frames apiece, one every
``--stride`` simulation ticks). Late in a generation only stalled cars remain,
so the rest of each generation is skipped.
"""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame
from PIL import Image

from racing_lab.app import App


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3, help="generations to fast-forward first")
    parser.add_argument("--generations", type=int, default=3, help="generations to record")
    parser.add_argument("--frames", type=int, default=26, help="frames recorded per generation")
    parser.add_argument("--stride", type=int, default=8, help="simulation ticks between frames")
    parser.add_argument("--width", type=int, default=800, help="output width in pixels")
    parser.add_argument("--colors", type=int, default=96)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent / "assets" / "evolution.gif")
    args = parser.parse_args()

    app = App()
    app.select_algorithm("evolution")
    trainer = app.trainer

    def next_generation():
        """Tick until the trainer starts a new generation."""
        before = len(trainer.history)
        while len(trainer.history) == before:
            trainer.tick()

    while len(trainer.history) < args.warmup:
        trainer.tick()
    images = []
    for _ in range(args.generations):
        for _ in range(args.frames):
            for _ in range(args.stride):
                trainer.tick()
            app.draw()
            size = app.screen.get_size()
            raw = pygame.image.tobytes(app.screen, "RGB")
            image = Image.frombytes("RGB", size, raw)
            height = round(size[1] * args.width / size[0])
            images.append(image.resize((args.width, height), Image.LANCZOS)
                          .quantize(colors=args.colors, method=Image.MEDIANCUT, dither=Image.NONE))
        next_generation()
    pygame.quit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.output, save_all=True, append_images=images[1:], optimize=True,
                   duration=round(1000 / args.fps), loop=0)
    print(f"{args.output} ({args.output.stat().st_size / 1e6:.2f} MB, {len(images)} frames, "
          f"{len(trainer.history)} generations)")


if __name__ == "__main__":
    main()
