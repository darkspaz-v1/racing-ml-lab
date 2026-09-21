"""Shared observations, car physics, rewards, and scoring for both learners."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .track import Track


SENSOR_ANGLES = (-90, -50, -25, 0, 25, 50, 90)
ACTION_NAMES = ("Left + gas", "Straight + gas", "Right + gas", "Coast", "Brake")
INPUT_NAMES = ("Ray -90", "Ray -50", "Ray -25", "Ray 0", "Ray +25",
               "Ray +50", "Ray +90", "Speed", "Goal sin", "Goal cos",
               "Lane offset", "Route angle")
MAX_SPEED = 5.2
MAX_RAYS = 15
# Optional non-ray inputs, in observation order: key -> (input names, value range).
EXTRA_INPUTS = {
    "speed": (("Speed",), (0.0, 1.0)),
    "goal": (("Goal sin", "Goal cos"), (-1.0, 1.0)),
    "lane": (("Lane offset",), (-1.0, 1.0)),
    "route": (("Route angle",), (-1.0, 1.0)),
}
EXTRA_LABELS = {"speed": "Speed", "goal": "Goal direction",
                "lane": "Lane offset", "route": "Route angle"}


def sensor_angles(rays: int) -> tuple[float, ...]:
    """Ray directions in degrees, spread evenly across the forward half-circle.

    Seven rays keep the original hand-tuned fan so the documented baseline
    experiment still reproduces exactly.
    """
    if rays == len(SENSOR_ANGLES):
        return SENSOR_ANGLES
    if rays == 1:
        return (0,)
    return tuple(round(-90 + 180 * i / (rays - 1), 1) for i in range(rays))


@dataclass(frozen=True)
class Sensors:
    """What the car can perceive: how many rays, and which extra inputs."""

    rays: int = len(SENSOR_ANGLES)
    extras: tuple[str, ...] = tuple(EXTRA_INPUTS)

    def __post_init__(self):
        if not 0 <= self.rays <= MAX_RAYS:
            raise ValueError(f"Use between 0 and {MAX_RAYS} rays")
        unknown = set(self.extras) - set(EXTRA_INPUTS)
        if unknown:
            raise ValueError(f"Unknown sensor input: {sorted(unknown)[0]}")
        # One canonical order, so equal setups compare and serialise equally.
        object.__setattr__(self, "extras", tuple(key for key in EXTRA_INPUTS if key in self.extras))
        if self.size == 0:
            raise ValueError("The car needs at least one input")

    @property
    def angles(self) -> tuple[float, ...]:
        return sensor_angles(self.rays) if self.rays else ()

    @property
    def names(self) -> tuple[str, ...]:
        rays = tuple("Ray 0" if a == 0 else f"Ray {a:+g}" for a in self.angles)
        return rays + tuple(name for key in self.extras for name in EXTRA_INPUTS[key][0])

    @property
    def size(self) -> int:
        return self.rays + sum(len(EXTRA_INPUTS[key][0]) for key in self.extras)

    def bounds(self, index: int) -> tuple[float, float]:
        if index < self.rays:
            return (0.0, 1.0)
        index -= self.rays
        for key in self.extras:
            names, value_range = EXTRA_INPUTS[key]
            if index < len(names):
                return value_range
            index -= len(names)
        raise IndexError("Input index is outside this sensor setup")

    def short_label(self) -> str:
        """Compact name for tables, e.g. "7 rays + all 4 inputs (12)"."""
        rays = f"{self.rays} ray{'' if self.rays == 1 else 's'}"
        if len(self.extras) == len(EXTRA_INPUTS):
            extras = " + all 4 inputs"
        elif self.extras:
            extras = " + " + ", ".join(EXTRA_LABELS[key].split()[0].lower() for key in self.extras)
        else:
            extras = " only"
        return f"{rays}{extras} ({self.size})"

    def label(self) -> str:
        extras = ", ".join(EXTRA_LABELS[key] for key in self.extras) or "no extra inputs"
        rays = f"{self.rays} ray{'' if self.rays == 1 else 's'}"
        return f"{rays} + {extras} ({self.size} input{'' if self.size == 1 else 's'})"


DEFAULT_SENSORS = Sensors()


@dataclass
class StepResult:
    reward: float
    crashed: bool = False
    checkpoint: bool = False
    lap: bool = False
    done: bool = False


@dataclass
class Car:
    track: Track
    max_steps: int = 750
    sensors: Sensors = DEFAULT_SENSORS
    x: float = field(init=False)
    y: float = field(init=False)
    angle: float = field(init=False)
    speed: float = 0.0
    steps: int = 0
    next_gate: int = 1
    passed_gates: int = 0
    laps: int = 0
    max_progress: float = 0.0
    crashed: bool = False
    done: bool = False
    lap_times: list[int] = field(default_factory=list)
    last_lap_step: int = 0

    def __post_init__(self):
        self.x, self.y, self.angle = self.track.start()
        self.max_progress = self.track.nearest_progress(self.x, self.y)

    def corners(self) -> list[tuple[float, float]]:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return [(self.x + forward * c - side * s,
                 self.y + forward * s + side * c)
                for forward, side in ((13, 8), (13, -8), (-13, 8), (-13, -8))]

    def observation(self) -> np.ndarray:
        values = [distance / 190 for distance in self.track.ray_distances(
            self.x, self.y, [self.angle + math.radians(degrees) for degrees in self.sensors.angles])]
        extras = self.sensors.extras
        if "speed" in extras:
            values.append(self.speed / MAX_SPEED)
        if "goal" in extras:
            gx, gy, _, _, _ = self.track.checkpoints[self.next_gate]
            bearing = math.atan2(gy - self.y, gx - self.x) - self.angle
            values += [math.sin(bearing), math.cos(bearing)]
        if "lane" in extras or "route" in extras:
            _, tx, ty, lateral = self.track.nearest_state(self.x, self.y)
            if "lane" in extras:
                values.append(max(-1, min(1, lateral / (self.track.width / 2))))
            if "route" in extras:
                values.append(math.sin(math.atan2(ty, tx) - self.angle))
        return np.asarray(values, dtype=np.float32)

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(0, crashed=self.crashed, done=True)
        if action not in range(len(ACTION_NAMES)):
            raise ValueError("Unknown driving action")
        before = (self.x, self.y)
        self.steps += 1
        if action in (0, 1, 2):
            self.speed = min(MAX_SPEED, self.speed + 0.20)
        elif action == 4:
            self.speed = max(0, self.speed - 0.30)
        else:
            self.speed *= 0.982
        self.speed *= 0.991
        turn = (-1 if action == 0 else 1 if action == 2 else 0)
        self.angle += turn * 0.063 * min(1, self.speed / 2.4)
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        crashed = not all(self.track.road_at(x, y) for x, y in self.corners())
        checkpoint = False
        lap = False
        if not crashed and self.track.gate_crossed(self.next_gate, before, (self.x, self.y)):
            checkpoint = True
            self.passed_gates += 1
            if self.next_gate == 0:
                self.laps += 1
                lap = True
                self.lap_times.append(self.steps - self.last_lap_step)
                self.last_lap_step = self.steps
            self.next_gate = (self.next_gate + 1) % len(self.track.checkpoints)

        # Shaping is paid only for a new furthest position in the ordered-gate
        # interval. Oscillation, reversing, or camping cannot collect it twice.
        raw = self.track.nearest_progress(self.x, self.y)
        absolute = self.laps * self.track.total_length + raw
        ceiling = self.track.next_gate_distance(self.next_gate, self.laps)
        absolute = min(absolute, ceiling)
        new_progress = max(self.max_progress, absolute) if not crashed else self.max_progress
        delta = max(0, new_progress - self.max_progress)
        self.max_progress = new_progress
        reward = delta * 0.025 + (4 if checkpoint else 0) + (40 if lap else 0) - 0.012
        if crashed:
            reward -= 8
        self.crashed = crashed
        self.done = crashed or self.steps >= self.max_steps
        return StepResult(reward, crashed, checkpoint, lap, self.done)

    def score(self) -> float:
        """Evaluation score shared by evolution and DQN, independent of reward."""
        return (self.max_progress / self.track.total_length * 100
                + 100 * self.laps - 0.015 * self.steps - (5 if self.crashed else 0))

    def snapshot(self, observation=None, hidden=None, outputs=None, action=None,
                 decision_pose=None) -> dict:
        decision_x, decision_y, decision_angle = (decision_pose if decision_pose is not None else
                                                   (self.x, self.y, self.angle))
        return {"x": round(self.x, 3), "y": round(self.y, 3),
                "angle": round(self.angle, 5), "speed": round(self.speed, 3),
                "decision_x": round(decision_x, 3), "decision_y": round(decision_y, 3),
                "decision_angle": round(decision_angle, 5),
                "step": self.steps, "gate": self.next_gate, "laps": self.laps,
                "progress": round(self.max_progress, 3), "crashed": self.crashed,
                "observation": [] if observation is None else np.asarray(observation).round(4).tolist(),
                "hidden": [] if hidden is None else np.asarray(hidden).round(4).tolist(),
                "outputs": [] if outputs is None else np.asarray(outputs).round(4).tolist(),
                "action": action}
