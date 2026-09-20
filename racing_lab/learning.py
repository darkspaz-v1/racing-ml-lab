"""Population evolution and replay-buffer DQN over the identical simulation."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from .network import Network
from .simulation import Car
from .track import Track


@dataclass
class Settings:
    seed: int = 7
    max_steps: int = 750
    population: int = 24
    mutation: float = 0.22
    learning_rate: float = 0.001
    epsilon_decay: float = 0.997
    batch_size: int = 32

    def as_dict(self) -> dict:
        return asdict(self)


class EvolutionTrainer:
    mode = "evolution"

    def __init__(self, track: Track, settings: Settings):
        self.track, self.settings = track, settings
        self.rng = np.random.default_rng(settings.seed)
        self.population = [Network(self.rng) for _ in range(settings.population)]
        self.generation = 1
        self.index = 0
        self.results: list[tuple[float, Network, bool, bool, float | None, float]] = []
        self.history: list[dict] = []
        self.car = Car(track, settings.max_steps)
        self.frames = [self.car.snapshot()]
        self.last_replay: dict | None = None
        self.best_replay: dict | None = None
        self.best_network: Network | None = None
        self.best_origin: dict | None = None
        self.best_score = -float("inf")
        self.last_observation = None
        self.last_hidden = None
        self.last_outputs = None
        self.last_action = None
        self.last_reward = 0.0

    @property
    def current_network(self) -> Network:
        return self.population[self.index]

    def tick(self) -> None:
        observation = self.car.observation()
        hidden, outputs = self.current_network.forward(observation)
        action = int(np.argmax(outputs))
        result = self.car.step(action)
        self.last_observation, self.last_hidden = observation, hidden
        self.last_outputs, self.last_action = outputs, action
        self.last_reward = result.reward
        self.frames.append(self.car.snapshot(observation, hidden, outputs, action))
        if result.done:
            self._finish_candidate()

    def _finish_candidate(self) -> None:
        score = self.car.score()
        lap_time = min(self.car.lap_times) if self.car.lap_times else None
        self.results.append((score, self.current_network.copy(), self.car.laps > 0,
                             self.car.crashed, lap_time,
                             min(1.0, self.car.max_progress / self.track.total_length)))
        replay = {"mode": self.mode, "run": self.generation,
                  "label": f"Generation {self.generation}, car {self.index + 1}",
                  "score": score, "frames": self.frames}
        self.last_replay = replay
        if score > self.best_score:
            self.best_score = score
            self.best_network = self.current_network.copy()
            self.best_origin = self.track.definition()
            self.best_replay = replay
        self.index += 1
        if self.index >= len(self.population):
            self._next_generation()
        self.car = Car(self.track, self.settings.max_steps)
        self.frames = [self.car.snapshot()]

    def _next_generation(self) -> None:
        ranked = sorted(self.results, key=lambda row: row[0], reverse=True)
        scores = [row[0] for row in ranked]
        lap_times = [row[4] for row in ranked if row[4] is not None]
        self.history.append({"run": self.generation, "best": max(scores),
                             "mean": float(np.mean(scores)),
                             "completion": sum(row[2] for row in ranked) / len(ranked),
                             "crashes": sum(row[3] for row in ranked) / len(ranked),
                             "best_lap": min(lap_times) if lap_times else None,
                             "fitness": max(scores), "reward": None,
                             "progress": max(row[5] for row in ranked)})
        parent_pool = [row[1] for row in ranked[:max(2, len(ranked) // 4)]]
        next_population = [ranked[0][1].copy(), ranked[1][1].copy()]
        while len(next_population) < self.settings.population:
            parent = parent_pool[int(self.rng.integers(len(parent_pool)))]
            next_population.append(parent.mutate(self.rng, self.settings.mutation))
        self.population = next_population
        self.results = []
        self.index = 0
        self.generation += 1


class DQNTrainer:
    mode = "dqn"

    def __init__(self, track: Track, settings: Settings):
        self.track, self.settings = track, settings
        self.rng = np.random.default_rng(settings.seed)
        self.online = Network(self.rng)
        self.target = self.online.copy()
        self.memory: deque = deque(maxlen=20000)
        self.episode = 1
        self.epsilon = 1.0
        self.training_steps = 0
        self.history: list[dict] = []
        self.evaluation_history: list[dict] = []
        self.car = Car(track, settings.max_steps)
        self.frames = [self.car.snapshot()]
        self.episode_reward = 0.0
        self.last_replay: dict | None = None
        self.best_replay: dict | None = None
        self.best_network: Network | None = None
        self.best_origin: dict | None = None
        self.best_score = -float("inf")
        self.last_observation = None
        self.last_hidden = None
        self.last_outputs = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_loss = None
        self.explored = False

    @property
    def current_network(self) -> Network:
        return self.online

    def tick(self) -> None:
        observation = self.car.observation()
        hidden, q_values = self.online.forward(observation)
        self.explored = bool(self.rng.random() < self.epsilon)
        action = int(self.rng.integers(5)) if self.explored else int(np.argmax(q_values))
        result = self.car.step(action)
        next_observation = self.car.observation()
        self.memory.append((observation, action, result.reward, next_observation, result.done))
        self.episode_reward += result.reward
        self.last_observation, self.last_hidden = observation, hidden
        self.last_outputs, self.last_action = q_values, action
        self.last_reward = result.reward
        self.frames.append(self.car.snapshot(observation, hidden, q_values, action))
        self.training_steps += 1
        if len(self.memory) >= max(128, self.settings.batch_size) and self.training_steps % 4 == 0:
            self._learn()
        if self.training_steps % 400 == 0:
            self.target = self.online.copy()
        if result.done:
            self._finish_episode()

    def _learn(self) -> None:
        indices = self.rng.choice(len(self.memory), self.settings.batch_size, replace=False)
        batch = [self.memory[int(i)] for i in indices]
        states = np.stack([row[0] for row in batch])
        actions = np.asarray([row[1] for row in batch], dtype=np.int64)
        rewards = np.asarray([row[2] for row in batch], dtype=np.float32)
        next_states = np.stack([row[3] for row in batch])
        done = np.asarray([row[4] for row in batch], dtype=np.float32)
        targets = rewards + 0.97 * (1 - done) * self.target.forward(next_states)[1].max(axis=1)
        self.last_loss = self.online.train_dqn(states, actions, targets, self.settings.learning_rate)

    def _finish_episode(self) -> None:
        score = self.car.score()
        lap_time = min(self.car.lap_times) if self.car.lap_times else None
        self.history.append({"run": self.episode, "best": score,
                             "mean": score, "completion": int(self.car.laps > 0),
                             "crashes": int(self.car.crashed),
                             "best_lap": lap_time, "fitness": score,
                             "reward": self.episode_reward,
                             "progress": min(1.0, self.car.max_progress / self.track.total_length),
                             "eval_score": None, "eval_completion": None})
        replay = {"mode": self.mode, "run": self.episode,
                  "label": f"Exploratory episode {self.episode}", "score": score,
                  "frames": self.frames}
        self.last_replay = replay
        if self.episode % 10 == 0:
            self._evaluate_greedy()
        self.epsilon = max(0.05, self.epsilon * self.settings.epsilon_decay)
        self.episode += 1
        self.car = Car(self.track, self.settings.max_steps)
        self.frames = [self.car.snapshot()]
        self.episode_reward = 0.0

    def _evaluate_greedy(self) -> None:
        """Measure the learned policy without ε exploration or weight updates."""
        car = Car(self.track, self.settings.max_steps)
        frames = [car.snapshot()]
        while not car.done:
            observation = car.observation()
            hidden, outputs = self.online.forward(observation)
            action = int(np.argmax(outputs))
            car.step(action)
            frames.append(car.snapshot(observation, hidden, outputs, action))
        score = car.score()
        self.evaluation_history.append({"run": self.episode, "score": score,
                                        "completion": int(car.laps > 0),
                                        "progress": min(1.0, car.max_progress / self.track.total_length),
                                        "crashed": int(car.crashed),
                                        "best_lap": min(car.lap_times) if car.lap_times else None})
        self.history[-1]["eval_score"] = score
        self.history[-1]["eval_completion"] = int(car.laps > 0)
        if score > self.best_score:
            self.best_score = score
            self.best_network = self.online.copy()
            self.best_origin = self.track.definition()
            self.best_replay = {"mode": self.mode, "run": self.episode,
                                "label": f"Greedy evaluation after episode {self.episode}",
                                "score": score, "frames": frames}
