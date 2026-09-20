"""Editable closed-loop tracks, road occupancy, and ordered checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np


WORLD_W, WORLD_H = 900, 640


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _segments_intersect(a, b, c, d) -> bool:
    ab = (b[0] - a[0], b[1] - a[1])
    cd = (d[0] - c[0], d[1] - c[1])
    den = _cross(ab, cd)
    if abs(den) < 1e-9:
        return False
    ac = (c[0] - a[0], c[1] - a[1])
    t, u = _cross(ac, cd) / den, _cross(ac, ab) / den
    return 0 < t < 1 and 0 < u < 1


@dataclass
class Track:
    points: list[tuple[float, float]]
    width: float = 92.0
    name: str = "Foundry Loop"
    gate_fractions: list[float] | None = None
    mask: np.ndarray = field(init=False, repr=False)
    checkpoints: list[tuple[float, float, float, float, float]] = field(init=False)
    lengths: list[float] = field(init=False)
    total_length: float = field(init=False)

    def __post_init__(self) -> None:
        self.points = [(float(x), float(y)) for x, y in self.points]
        self.validate()
        self.lengths = [math.dist(a, b) for a, b in self.segments]
        self.total_length = sum(self.lengths)
        self.mask = self._make_mask()
        # Gates at equal distance along the road. Gate 0 is the finish line.
        if self.gate_fractions is None:
            count = max(8, round(self.total_length / 125))
            self.gate_fractions = [i / count for i in range(count)]
        else:
            self.gate_fractions = sorted(float(value) for value in self.gate_fractions)
            if (len(self.gate_fractions) < 4 or self.gate_fractions[0] != 0
                    or any(b - a < 0.015 for a, b in zip(self.gate_fractions, self.gate_fractions[1:]))
                    or self.gate_fractions[-1] >= 0.985):
                raise ValueError("Use at least four ordered gates, including the start gate at 0.")
        self.checkpoints = [self._sample(self.total_length * value) for value in self.gate_fractions]

    @property
    def segments(self):
        return list(zip(self.points, self.points[1:] + self.points[:1]))

    def validate(self) -> None:
        if len(self.points) < 5:
            raise ValueError("Add at least five centerline points before closing the track.")
        if not 55 <= self.width <= 160:
            raise ValueError("Track width must be between 55 and 160 pixels.")
        margin = self.width / 2 + 12
        if any(not (margin <= x <= WORLD_W - margin and margin <= y <= WORLD_H - margin)
               for x, y in self.points):
            raise ValueError("Keep every centerline point inside the canvas margin.")
        segs = self.segments
        if any(math.dist(a, b) < 35 for a, b in segs):
            raise ValueError("Centerline points must be at least 35 pixels apart.")
        n = len(segs)
        for i in range(n):
            for j in range(i + 1, n):
                if j in (i + 1, n - 1 if i == 0 else -1):
                    continue
                if _segments_intersect(*segs[i], *segs[j]):
                    raise ValueError("The centerline crosses itself; use a single loop.")

    def _make_mask(self) -> np.ndarray:
        mask = np.zeros((WORLD_H, WORLD_W), dtype=bool)
        radius2 = (self.width / 2) ** 2
        for (ax, ay), (bx, by) in self.segments:
            pad = self.width / 2 + 2
            x0, x1 = max(0, int(min(ax, bx) - pad)), min(WORLD_W, int(max(ax, bx) + pad + 1))
            y0, y1 = max(0, int(min(ay, by) - pad)), min(WORLD_H, int(max(ay, by) + pad + 1))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dx, dy = bx - ax, by - ay
            t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / (dx * dx + dy * dy), 0, 1)
            dist2 = (xx - (ax + t * dx)) ** 2 + (yy - (ay + t * dy)) ** 2
            mask[y0:y1, x0:x1] |= dist2 <= radius2
        return mask

    def road_at(self, x: float, y: float) -> bool:
        ix, iy = int(round(x)), int(round(y))
        return 0 <= ix < WORLD_W and 0 <= iy < WORLD_H and bool(self.mask[iy, ix])

    def ray_distance(self, x: float, y: float, angle: float, max_distance: float = 190) -> float:
        c, s = math.cos(angle), math.sin(angle)
        previous = 0.0
        for distance in range(6, int(max_distance) + 7, 6):
            distance = min(distance, max_distance)
            if not self.road_at(x + c * distance, y + s * distance):
                lo, hi = previous, distance
                for _ in range(4):
                    middle = (lo + hi) / 2
                    if self.road_at(x + c * middle, y + s * middle):
                        lo = middle
                    else:
                        hi = middle
                return hi
            previous = distance
        return max_distance

    def _sample(self, distance: float):
        remaining = distance
        for ((ax, ay), (bx, by)), length in zip(self.segments, self.lengths):
            if remaining <= length:
                t = remaining / length
                return (ax + (bx - ax) * t, ay + (by - ay) * t,
                        (bx - ax) / length, (by - ay) / length, distance)
            remaining -= length
        return self._sample(0)

    def nearest_progress(self, x: float, y: float) -> float:
        return self.nearest_state(x, y)[0]

    def nearest_state(self, x: float, y: float) -> tuple[float, float, float, float]:
        """Arc distance, tangent x/y, and signed distance from road center."""
        best_distance, best_progress = float("inf"), 0.0
        best_tangent = (1.0, 0.0)
        best_lateral = 0.0
        cumulative = 0.0
        for ((ax, ay), (bx, by)), length in zip(self.segments, self.lengths):
            dx, dy = bx - ax, by - ay
            t = max(0, min(1, ((x - ax) * dx + (y - ay) * dy) / (length * length)))
            distance = (x - ax - t * dx) ** 2 + (y - ay - t * dy) ** 2
            if distance < best_distance:
                best_distance, best_progress = distance, cumulative + t * length
                best_tangent = (dx / length, dy / length)
                best_lateral = ((x - ax - t * dx) * -dy + (y - ay - t * dy) * dx) / length
            cumulative += length
        return best_progress, *best_tangent, best_lateral

    def next_gate_distance(self, index: int, lap: int) -> float:
        return (lap + (1 if index == 0 else 0)) * self.total_length + self.checkpoints[index][4]

    def gate_crossed(self, index: int, before: tuple[float, float], after: tuple[float, float]) -> bool:
        x, y, tx, ty, _ = self.checkpoints[index]
        previous = (before[0] - x) * tx + (before[1] - y) * ty
        current = (after[0] - x) * tx + (after[1] - y) * ty
        if not previous < 0 <= current:
            return False
        proportion = -previous / (current - previous)
        cx = before[0] + proportion * (after[0] - before[0])
        cy = before[1] + proportion * (after[1] - before[1])
        return abs((cx - x) * -ty + (cy - y) * tx) <= self.width / 2

    def start(self) -> tuple[float, float, float]:
        x, y, tx, ty, _ = self.checkpoints[0]
        # Start just after the finish gate, so the next gate is number 1.
        return x + tx * 13, y + ty * 13, math.atan2(ty, tx)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.definition(), indent=2), encoding="utf-8")

    def definition(self) -> dict:
        return {"name": self.name, "width": self.width, "points": self.points,
                "gate_fractions": self.gate_fractions}

    @classmethod
    def load(cls, path: str | Path) -> "Track":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["points"], float(data["width"]), data.get("name", Path(path).stem),
                   data.get("gate_fractions"))


def default_track() -> Track:
    return Track([(450, 112), (590, 122), (735, 205), (760, 350),
                  (690, 475), (530, 520), (370, 510), (220, 470),
                  (130, 345), (160, 200), (290, 125)], 100, "Foundry Loop")
