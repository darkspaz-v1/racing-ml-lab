import json
import math
import random

import numpy as np
import pygame
import pytest

from racing_lab import app as app_module
from racing_lab.app import App
from racing_lab.decision_lab import DecisionProbe
from racing_lab.learning import DQNTrainer, EvolutionTrainer, Settings
from racing_lab.network import Network
from racing_lab.simulation import (EXTRA_INPUTS, INPUT_NAMES, MAX_RAYS, SENSOR_ANGLES, Car,
                                   Sensors, sensor_angles)
from racing_lab.track import Track, default_track


@pytest.fixture
def workbench(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    lab = App()
    yield lab
    pygame.quit()


def reference_ray(track, x, y, angle, max_distance=190):
    """The original one-ray-at-a-time cast, kept as the definition of correct."""
    c, s = math.cos(angle), math.sin(angle)
    previous = 0.0
    for distance in range(6, int(max_distance) + 7, 6):
        distance = min(distance, max_distance)
        if not track.road_at(x + c * distance, y + s * distance):
            lo, hi = previous, distance
            for _ in range(4):
                middle = (lo + hi) / 2
                if track.road_at(x + c * middle, y + s * middle):
                    lo = middle
                else:
                    hi = middle
            return hi
        previous = distance
    return max_distance


def test_default_sensors_are_the_original_twelve_inputs():
    sensors = Sensors()
    assert sensors.names == INPUT_NAMES
    assert sensors.angles == SENSOR_ANGLES
    assert sensors.size == 12
    assert [sensors.bounds(i) for i in range(12)] == [(0.0, 1.0)] * 8 + [(-1.0, 1.0)] * 4


def test_sensor_setups_size_names_and_bounds_agree():
    for rays in range(MAX_RAYS + 1):
        for mask in range(16):
            extras = tuple(key for bit, key in enumerate(EXTRA_INPUTS) if mask >> bit & 1)
            if rays == 0 and not extras:
                continue
            sensors = Sensors(rays, extras)
            assert len(sensors.names) == sensors.size == len(set(sensors.names))
            assert len(sensors.angles) == rays
            for i in range(sensors.size):
                assert sensors.bounds(i) in ((0.0, 1.0), (-1.0, 1.0))
            with pytest.raises(IndexError):
                sensors.bounds(sensors.size)


def test_ray_angles_are_symmetric_and_span_the_forward_half_circle():
    assert sensor_angles(1) == (0,)
    for rays in range(2, MAX_RAYS + 1):
        angles = sensor_angles(rays)
        assert angles[0] == -90 and angles[-1] == 90
        assert list(angles) == sorted(angles)
        assert all(abs(a + b) < 1e-9 for a, b in zip(angles, reversed(angles)))


def test_extras_are_stored_in_one_canonical_order():
    assert Sensors(3, ("route", "speed")) == Sensors(3, ("speed", "route"))
    assert Sensors(3, ("route", "speed")).extras == ("speed", "route")


@pytest.mark.parametrize("kwargs, message", [
    (dict(rays=0, extras=()), "at least one input"),
    (dict(rays=MAX_RAYS + 1), "between 0"),
    (dict(rays=-1), "between 0"),
    (dict(rays=4, extras=("gps",)), "Unknown sensor input"),
])
def test_invalid_sensor_setups_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Sensors(**kwargs)


def test_batched_rays_match_the_original_single_ray_cast():
    track = default_track()
    rng = random.Random(11)
    checked = 0
    while checked < 400:
        x, y = rng.uniform(0, 900), rng.uniform(0, 640)
        if not track.road_at(x, y):
            continue
        angles = [rng.uniform(-math.pi, math.pi) for _ in range(9)]
        assert track.ray_distances(x, y, angles) == [reference_ray(track, x, y, a) for a in angles]
        checked += 1


def test_numpy_nearest_state_matches_the_loop_on_a_dense_track():
    dense = Track([(450 + 240 * math.cos(2 * math.pi * i / 32), 320 + 200 * math.sin(2 * math.pi * i / 32))
                   for i in range(32)], 90)
    rng = random.Random(5)
    for _ in range(2000):
        x, y = rng.uniform(0, 900), rng.uniform(0, 640)
        assert dense.nearest_state(x, y) == dense._nearest_loop(x, y)


@pytest.mark.parametrize("sensors", [Sensors(0), Sensors(1, ()), Sensors(15), Sensors(4, ("route",))])
def test_observations_match_the_sensor_setup_and_stay_in_range(sensors):
    car = Car(default_track(), sensors=sensors)
    for step in range(60):
        observation = car.observation()
        assert observation.shape == (sensors.size,)
        for i, value in enumerate(observation):
            low, high = sensors.bounds(i)
            assert low - 1e-6 <= value <= high + 1e-6
        car.step(1 if step % 5 else 2)


def test_default_observation_values_are_unchanged():
    car = Car(default_track())
    reference = Car(default_track(), sensors=Sensors())
    for _ in range(40):
        assert np.array_equal(car.observation(), reference.observation())
        car.step(1)
        reference.step(1)
    rays = [reference_ray(car.track, car.x, car.y, car.angle + math.radians(d)) / 190
            for d in SENSOR_ANGLES]
    assert np.allclose(car.observation()[:7], rays)


def test_trainers_build_networks_sized_for_the_sensor_setup():
    settings = Settings(rays=11, inputs=("speed",), population=6, max_steps=60)
    for trainer in (EvolutionTrainer(default_track(), settings), DQNTrainer(default_track(), settings)):
        assert trainer.current_network.w1.shape == (12, 16)
        for _ in range(80):
            trainer.tick()
        assert trainer.car.sensors == Sensors(11, ("speed",))


def test_settings_round_trip_through_json():
    settings = Settings(rays=5, inputs=["route", "speed"])
    assert settings.inputs == ("speed", "route")
    restored = Settings(**json.loads(json.dumps(settings.as_dict())))
    assert restored == settings and restored.sensors() == Sensors(5, ("speed", "route"))


def test_models_with_other_input_widths_load_and_malformed_ones_do_not(tmp_path):
    network = Network(np.random.default_rng(1), 19)
    network.save(tmp_path / "wide.npz", "evolution")
    loaded, _, _ = Network.load(tmp_path / "wide.npz")
    assert loaded.w1.shape == (19, 16)
    network.w2 = network.w2[:, :4]
    network.save(tmp_path / "bad.npz", "evolution")
    with pytest.raises(ValueError):
        Network.load(tmp_path / "bad.npz")


def test_decision_probe_uses_the_sensor_setup_for_size_and_bounds():
    sensors = Sensors(3, ("speed", "lane"))
    network = Network(np.random.default_rng(2), sensors.size)
    probe = DecisionProbe.capture(network, np.zeros(sensors.size), sensors=sensors)
    probe.set_input(3)  # speed
    probe.set_value(-5)
    assert probe.edited[3] == 0
    probe.set_input(4)  # lane offset
    probe.set_value(-5)
    assert probe.edited[4] == -1
    with pytest.raises(IndexError):
        probe.set_input(5)
    with pytest.raises(ValueError):
        DecisionProbe.capture(network, np.zeros(12), sensors=sensors)


def test_applying_a_sensor_setup_retrains_and_records_the_previous_result(workbench):
    workbench.settings.max_steps = 40
    workbench.reset_trainers()
    while not workbench.trainer.history:
        workbench.tick()
    workbench.apply_sensors(Sensors(9, ("speed",)))
    assert workbench.sensors == Sensors(9, ("speed",))
    assert workbench.trainer.generation == 1 and not workbench.trainer.history
    for trainer in workbench.trainers.values():
        assert trainer.current_network.w1.shape == (10, 16)
    assert workbench.sensor_results[-1]["label"] == "7 rays + all 4 inputs (12)"
    assert workbench.sensor_results[-1]["generations"] == 1
    workbench.apply_sensors(Sensors(9, ("speed",)))  # unchanged: no extra row
    assert len(workbench.sensor_results) == 1


def test_sensor_panel_keyboard_toggles_and_draft_edits(workbench):
    workbench.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_r}))
    assert workbench.sensor_panel
    workbench.change_draft_rays(+3)
    workbench.toggle_draft_input("goal")
    assert workbench.sensor_draft == Sensors(10, ("speed", "lane", "route"))
    assert workbench.sensors == Sensors()  # drafts do not apply themselves
    workbench.draw()
    workbench.apply_sensors()
    assert workbench.sensors.size == 13
    workbench.draw()
    workbench.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}))
    assert not workbench.sensor_panel


def test_draft_cannot_remove_the_last_input(workbench):
    workbench.toggle_sensor_panel()
    workbench.sensor_draft = Sensors(0, ("speed",))
    workbench.toggle_draft_input("speed")
    assert workbench.sensor_draft == Sensors(0, ("speed",))


def test_loading_a_model_adopts_its_sensor_setup(workbench, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MODELS", tmp_path)
    workbench.apply_sensors(Sensors(13, ("speed",)))
    workbench.trainer.best_network = workbench.trainer.current_network.copy()
    workbench.save_model()
    path = next(tmp_path.glob("*.npz"))
    workbench.apply_sensors(Sensors())
    monkeypatch.setattr(workbench, "_file_dialog", lambda *args: path)
    workbench.load_model()
    assert workbench.sensors == Sensors(13, ("speed",))
    assert workbench.trainer.population[0].w1.shape == (14, 16)
    workbench.start_race()
    for _ in range(5):
        workbench.tick()
    assert workbench.opponent.steps == 5


def test_every_inspector_view_draws_at_the_widest_and_narrowest_setups(workbench):
    for sensors in (Sensors(MAX_RAYS), Sensors(1, ())):
        workbench.apply_sensors(sensors)
        for _ in range(3):
            workbench.tick()
        workbench.draw()
        workbench.toggle_wiring()
        workbench.draw()
        workbench.open_sandbox()
        assert workbench.sandbox.sensors == sensors
        workbench.draw()
        workbench._close_sandbox()


def test_clicks_inside_overlays_do_not_select_a_car_underneath(workbench):
    workbench.tick()
    for opener in (workbench.toggle_wiring, workbench.toggle_sensor_panel):
        opener()
        workbench.draw()
        before = workbench.trainers["evolution"].index
        workbench.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (300, 620)}))
        assert workbench.trainers["evolution"].index == before
