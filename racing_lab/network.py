"""A small NumPy neural network whose forward and backward passes are visible."""

from __future__ import annotations

import json

import numpy as np


class Network:
    def __init__(self, rng: np.random.Generator, inputs=12, hidden=16, outputs=5):
        self.w1 = rng.normal(0, 1 / np.sqrt(inputs), (inputs, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, outputs)).astype(np.float32)
        self.b2 = np.zeros(outputs, dtype=np.float32)

    def copy(self) -> "Network":
        new = object.__new__(Network)
        for name in ("w1", "b1", "w2", "b2"):
            setattr(new, name, getattr(self, name).copy())
        return new

    def forward(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        hidden = np.tanh(x @ self.w1 + self.b1)
        outputs = hidden @ self.w2 + self.b2
        return hidden, outputs

    def act(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)[1]))

    def mutate(self, rng: np.random.Generator, scale: float) -> "Network":
        child = self.copy()
        for name in ("w1", "b1", "w2", "b2"):
            values = getattr(child, name)
            chosen = rng.random(values.shape) < 0.18
            values += chosen * rng.normal(0, scale, values.shape).astype(np.float32)
        return child

    def train_dqn(self, states: np.ndarray, actions: np.ndarray,
                  targets: np.ndarray, learning_rate: float) -> float:
        """One clipped-error DQN gradient step, written out for inspection."""
        hidden, q_values = self.forward(states)
        rows = np.arange(len(actions))
        errors = q_values[rows, actions] - targets
        selected_grad = np.clip(errors, -1, 1) / len(actions)
        output_grad = np.zeros_like(q_values)
        output_grad[rows, actions] = selected_grad
        grad_w2 = hidden.T @ output_grad
        grad_b2 = output_grad.sum(axis=0)
        hidden_grad = (output_grad @ self.w2.T) * (1 - hidden ** 2)
        grad_w1 = states.T @ hidden_grad
        grad_b1 = hidden_grad.sum(axis=0)
        for name, gradient in (("w1", grad_w1), ("b1", grad_b1),
                               ("w2", grad_w2), ("b2", grad_b2)):
            getattr(self, name)[:] -= learning_rate * np.clip(gradient, -5, 5)
        return float(np.mean(np.where(abs(errors) < 1,
                                      0.5 * errors ** 2, abs(errors) - 0.5)))

    def save(self, path, mode: str, metadata: dict | None = None) -> None:
        np.savez_compressed(path, mode=mode, metadata=json.dumps(metadata or {}),
                            w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path) -> tuple["Network", str, dict]:
        with np.load(path, allow_pickle=False) as data:
            net = object.__new__(cls)
            for name in ("w1", "b1", "w2", "b2"):
                setattr(net, name, data[name].astype(np.float32).copy())
            mode = str(data["mode"])
            metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
        # Input width depends on the sensor setup it was trained with; the hidden
        # layer and the five driving actions are fixed.
        if (net.w1.ndim != 2 or net.w1.shape[1] != 16 or not 1 <= net.w1.shape[0] <= 20
                or net.b1.shape != (16,) or net.w2.shape != (16, 5) or net.b2.shape != (5,)):
            raise ValueError("Model architecture does not match this version of the lab")
        return net, mode, metadata
