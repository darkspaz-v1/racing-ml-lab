import math

import numpy as np
import pytest

from racing_lab.learning import DQNTrainer, EvolutionTrainer, Settings
from racing_lab.network import Network
from racing_lab.simulation import Car
from racing_lab.track import Track, default_track


@pytest.fixture(scope="module")
def track():
    return default_track()


def test_track_round_trip_and_validation(track, tmp_path):
    path = tmp_path / "track.json"
    track.save(path)
    loaded = Track.load(path)
    assert loaded.points == track.points
    assert loaded.gate_fractions == track.gate_fractions
    assert loaded.road_at(*loaded.start()[:2])
    with pytest.raises(ValueError, match="five"):
        Track([(100, 100), (200, 100), (200, 200)])


def test_sensor_hits_side_wall(track):
    car = Car(track)
    ray = track.ray_distance(car.x, car.y, car.angle + math.pi / 2)
    assert 35 < ray < 70
    assert len(car.observation()) == 12
    assert np.all(np.isfinite(car.observation()))


def test_ordered_gates_and_lap(track):
    car = Car(track)
    x, y, tx, ty, _ = track.checkpoints[2]
    car.x, car.y, car.angle, car.speed = x - 2 * tx, y - 2 * ty, math.atan2(ty, tx), 5
    result = car.step(1)
    assert not result.checkpoint  # Gate 2 cannot substitute for gate 1.
    assert car.next_gate == 1

    x, y, tx, ty, _ = track.checkpoints[1]
    car = Car(track)
    car.x, car.y, car.angle, car.speed = x - 2 * tx, y - 2 * ty, math.atan2(ty, tx), 5
    result = car.step(1)
    assert result.checkpoint and not result.lap
    assert car.next_gate == 2

    car = Car(track)
    car.next_gate = 0
    car.passed_gates = len(track.checkpoints) - 1
    x, y, tx, ty, _ = track.checkpoints[0]
    car.x, car.y, car.angle, car.speed = x - 2 * tx, y - 2 * ty, math.atan2(ty, tx), 5
    result = car.step(1)
    assert result.lap and car.laps == 1 and car.next_gate == 1


def test_crash_and_no_repeat_progress_reward(track):
    car = Car(track)
    initial = car.max_progress
    car.step(1)
    car.x, car.y = track.start()[:2]
    car.speed = 0
    prior = car.max_progress
    car.step(3)
    assert car.max_progress >= prior >= initial
    assert car.max_progress - prior < 1  # Returning near start pays no new distance.

    car = Car(track)
    car.x, car.y = 10, 10
    result = car.step(1)
    assert result.crashed and result.done and result.reward < 0


def test_network_update_and_safe_save(tmp_path):
    rng = np.random.default_rng(9)
    net = Network(rng)
    before = net.w1.copy()
    states = rng.random((32, 12), dtype=np.float32)
    targets = np.ones(32, dtype=np.float32) * 3
    loss = net.train_dqn(states, np.zeros(32, dtype=np.int64), targets, 0.001)
    assert loss > 0
    assert not np.array_equal(before, net.w1)
    path = tmp_path / "driver.npz"
    net.save(path, "dqn", {"seed": 9})
    loaded, mode, metadata = Network.load(path)
    assert mode == "dqn" and metadata["seed"] == 9
    np.testing.assert_array_equal(loaded.forward(states)[1], net.forward(states)[1])


def test_trainers_change_their_networks_and_capture_replays(track):
    settings = Settings(seed=12, max_steps=70, population=8, batch_size=8)
    evolution = EvolutionTrainer(track, settings)
    original_population = [net.w1.copy() for net in evolution.population]
    while evolution.generation == 1:
        evolution.tick()
    assert len(evolution.history) == 1
    assert evolution.best_replay and evolution.best_replay["frames"]
    assert any(not np.array_equal(net.w1, original_population[i])
               for i, net in enumerate(evolution.population))

    dqn = DQNTrainer(track, settings)
    before = dqn.online.w1.copy()
    while not dqn.evaluation_history:
        dqn.tick()
    assert dqn.last_loss is not None
    assert not np.array_equal(before, dqn.online.w1)
    assert dqn.best_replay and dqn.best_replay["frames"]
    assert dqn.best_replay["label"].startswith("Greedy evaluation")


def test_seed_repeats_same_initial_driver(track):
    settings = Settings(seed=32)
    a, b = EvolutionTrainer(track, settings), EvolutionTrainer(track, settings)
    np.testing.assert_array_equal(a.current_network.w1, b.current_network.w1)
    for _ in range(10):
        a.tick()
        b.tick()
    assert a.car.snapshot() == b.car.snapshot()
