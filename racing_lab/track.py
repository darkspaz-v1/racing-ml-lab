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
        self._index_segments()
        self.mask = self._make_mask()
        self._flat_mask = self.mask.tobytes()
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

    def _index_segments(self) -> None:
        """Segment arrays for the vectorised nearest-point search."""
        segs = self.segments
        self._ax = np.array([a[0] for a, _ in segs])
        self._ay = np.array([a[1] for a, _ in segs])
        self._dx = np.array([b[0] - a[0] for a, b in segs])
        self._dy = np.array([b[1] - a[1] for a, b in segs])
        self._len = np.array(self.lengths)
        self._len2 = self._len * self._len
        cumulative, running = [], 0.0
        for length in self.lengths:
            cumulative.append(running)
            running += length
        self._cumulative = np.array(cumulative)
        self._segment_rows = [(a[0], a[1], b[0] - a[0], b[1] - a[1], length)
                              for (a, b), length in zip(segs, self.lengths)]

    def road_at(self, x: float, y: float) -> bool:
        ix, iy = int(round(x)), int(round(y))
        return 0 <= ix < WORLD_W and 0 <= iy < WORLD_H and bool(self.mask[iy, ix])

    def ray_distance(self, x: float, y: float, angle: float, max_distance: float = 190) -> float:
        return self.ray_distances(x, y, (angle,), max_distance)[0]

    def ray_distances(self, x: float, y: float, angles, max_distance: float = 190) -> list:
        """Distance to the road edge along each angle.

        Marches in 6 px steps until a sample leaves the road, then bisects that
        step four times. The road test is inlined against a flat byte copy of the
        mask because this loop is most of the simulation's run time.
        """
        flat, width, height = self._flat_mask, WORLD_W, WORLD_H
        steps = [min(d, max_distance) for d in range(6, int(max_distance) + 7, 6)]
        results = []
        for angle in angles:
            c, s = math.cos(angle), math.sin(angle)
            previous, result = 0.0, max_distance
            for distance in steps:
                ix, iy = round(x + c * distance), round(y + s * distance)
                if not (0 <= ix < width and 0 <= iy < height and flat[iy * width + ix]):
                    lo, hi = previous, distance
                    for _ in range(4):
                        middle = (lo + hi) / 2
                        ix, iy = round(x + c * middle), round(y + s * middle)
                        if 0 <= ix < width and 0 <= iy < height and flat[iy * width + ix]:
                            lo = middle
                        else:
                            hi = middle
                    result = hi
                    break
                previous = distance
            results.append(result)
        return results

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
        # A plain loop is faster for a few segments; NumPy wins on dense circuits.
        # Both evaluate the same expressions in the same order, so results match.
        if len(self._segment_rows) < 32:
            return self._nearest_loop(x, y)
        t = np.maximum(0, np.minimum(1, ((x - self._ax) * self._dx + (y - self._ay) * self._dy)
                                     / self._len2))
        ox = x - self._ax - t * self._dx
        oy = y - self._ay - t * self._dy
        i = int(np.argmin(ox ** 2 + oy ** 2))
        length = self._len[i]
        return (float(self._cumulative[i] + t[i] * length),
                float(self._dx[i] / length), float(self._dy[i] / length),
                float((ox[i] * -self._dy[i] + oy[i] * self._dx[i]) / length))

    def _nearest_loop(self, x: float, y: float) -> tuple[float, float, float, float]:
        best_distance, best_progress = float("inf"), 0.0
        best_tangent = (1.0, 0.0)
        best_lateral = 0.0
        cumulative = 0.0
        for ax, ay, dx, dy, length in self._segment_rows:
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


def apex_circuit() -> Track:
    """A motor-racing-style layout: a long main straight, sweeping corners, and a tight U-turn.

    The centreline is a closed Catmull-Rom spline through 29 control points,
    resampled every ~37 px so corners are smooth rather than polygonal.
    """
    return Track([
        (300, 108), (337.1, 107.6), (374.2, 107), (411.4, 106.5),
        (448.5, 106.1), (485.6, 106), (522.7, 105.6), (559.9, 105.4),
        (597, 105.9), (634.1, 107.6), (671.1, 109.8), (708.2, 112.3),
        (744.6, 119.3), (776.5, 137.7), (800.8, 165.5), (813.7, 200),
        (809.5, 236.7), (794.7, 270.5), (768.1, 296.3), (745.3, 325),
        (744.7, 361.8), (755.2, 397.3), (776, 427.6), (799.9, 455.8),
        (801, 492.3), (783.4, 524.3), (748.9, 536.9), (712, 541),
        (675, 543.8), (637.9, 544.1), (601.6, 536.7), (572.8, 513.9),
        (554, 482.2), (550.2, 445.4), (532.2, 413.1), (501, 394.4),
        (464.1, 391.8), (428.2, 400.8), (401.2, 424.8), (392.3, 460.7),
        (387.3, 497.4), (363.9, 525.8), (331, 542.2), (294.1, 546.4),
        (257, 546.3), (219.9, 545.9), (182.8, 544.3), (148, 532.6),
        (123.8, 504.8), (109.3, 470.8), (106.3, 434), (112.4, 397.4),
        (124.7, 362.5), (138.8, 328.3), (133.6, 292.2), (117.8, 258.6),
        (110.1, 222.4), (112.1, 185.6), (128.6, 152.5), (154.6, 126.3),
        (189.2, 113.3), (225.8, 107.5), (262.9, 106.9),
    ], 88, "Apex Circuit")
