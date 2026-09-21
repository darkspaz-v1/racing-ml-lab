"""A frozen, model-only what-if probe for teaching neural-network decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import Network
from .simulation import DEFAULT_SENSORS, Sensors


@dataclass
class DecisionProbe:
    network: Network
    original: np.ndarray
    edited: np.ndarray
    actual_action: int | None
    sensors: Sensors = DEFAULT_SENSORS
    selected_input: int = 0

    @classmethod
    def capture(cls, network: Network, observation, actual_action=None,
                sensors: Sensors = DEFAULT_SENSORS) -> "DecisionProbe":
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (sensors.size,) or network.w1.shape[0] != sensors.size:
            raise ValueError(f"A decision probe needs all {sensors.size} sensor inputs")
        return cls(network.copy(), values.copy(), values.copy(), actual_action, sensors)

    def bounds(self) -> tuple[float, float]:
        return self.sensors.bounds(self.selected_input)

    def set_input(self, index: int) -> None:
        if not 0 <= index < self.sensors.size:
            raise IndexError(f"Input index must be between 0 and {self.sensors.size - 1}")
        self.selected_input = index

    def set_value(self, value: float) -> None:
        low, high = self.bounds()
        self.edited[self.selected_input] = np.clip(value, low, high)

    def reset(self) -> None:
        self.edited[:] = self.original

    def evaluate(self) -> tuple[np.ndarray, np.ndarray, int]:
        hidden, outputs = self.network.forward(self.edited)
        return hidden, outputs, int(np.argmax(outputs))

    def baseline(self) -> tuple[np.ndarray, np.ndarray, int]:
        hidden, outputs = self.network.forward(self.original)
        return hidden, outputs, int(np.argmax(outputs))
