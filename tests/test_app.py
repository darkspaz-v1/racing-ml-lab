import math

import numpy as np
import pygame
import pytest

from racing_lab import app as app_module
from racing_lab.app import App, RX
from racing_lab.simulation import SENSOR_ANGLES
from racing_lab.track import Track


@pytest.fixture
def workbench(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    lab = App()
    yield lab
    pygame.quit()


def test_editor_applies_new_track_and_renders(workbench, tmp_path):
    workbench.start_editor()
    workbench.editor_points[1] = (605, 132)
    workbench.editor_gates = []
    workbench.apply_editor()
    assert workbench.view == "train"
    assert workbench.track.points[1] == (605, 132)
    assert workbench.trainer.generation == 1
    workbench.tick()
    workbench.draw()
    pygame.image.save(workbench.screen, tmp_path / "workbench.png")
    assert (tmp_path / "workbench.png").stat().st_size > 1000


def test_saved_model_races_on_a_different_current_track(workbench, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "MODELS", tmp_path)
    workbench.trainer.best_network = workbench.trainer.current_network.copy()
    workbench.trainer.best_score = 10.0
    workbench.save_model()
    path = next(tmp_path.glob("*.npz"))
    workbench.track = Track.load(app_module.TRACKS / "harbor-loop.json")
    workbench.reset_trainers()
    monkeypatch.setattr(workbench, "_file_dialog", lambda *args: path)
    workbench.load_model()
    assert workbench.track.name == "Harbor Loop"
    assert workbench.trainer.best_network is not None
    assert workbench.trainer.best_origin["name"] == "Apex Circuit"
    assert workbench.trainer.best_score == -float("inf")
    workbench.start_race()
    for _ in range(5):
        workbench.tick()
    assert workbench.opponent.steps == 5


def test_replay_round_trip_and_step_controls(workbench, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "REPLAYS", tmp_path)
    workbench.settings.max_steps = 30
    workbench.reset_trainers()
    while workbench.trainer.last_replay is None:
        workbench.trainer.tick()
    workbench.replay_last()
    assert workbench.view == "replay" and workbench.paused
    workbench._step_replay(1)
    assert workbench.replay_index == 1
    workbench.save_replay()
    path = next(tmp_path.glob("*.json"))
    monkeypatch.setattr(workbench, "_file_dialog", lambda *args: path)
    workbench.open_replay()
    assert workbench.replay["frames"]
    assert workbench.track.name == "Apex Circuit"
    workbench.draw()


def test_compare_view_advances_both_algorithms_and_switches_inspector(workbench):
    workbench.start_compare()
    workbench.tick()
    evolution = workbench.trainers["evolution"]
    dqn = workbench.trainers["dqn"]
    assert all(car.steps == 1 for car in evolution.cars)
    assert dqn.car.steps == 1
    workbench.focus_evolution(2)
    assert evolution.index == 2
    assert workbench._world_frame() is evolution.traces[2][-1]
    workbench.algorithm = "dqn"
    assert workbench._world_frame() is dqn.frames[-1]
    workbench.draw()


def test_named_training_tabs_and_compare_inspector_are_clickable(workbench):
    workbench.draw()
    workbench.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                             {"button": 1, "pos": (420, 39)}))
    assert workbench.view == "train" and workbench.algorithm == "dqn"
    workbench.draw()
    workbench.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                             {"button": 1, "pos": (855, 39)}))
    assert workbench.view == "compare"
    workbench.algorithm = "evolution"
    workbench.draw()
    workbench.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                             {"button": 1, "pos": (RX + 460, 104)}))
    assert workbench.algorithm == "dqn"


def test_metric_selector_cycles_valid_metrics_in_each_training_mode(workbench):
    workbench.draw()
    workbench.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                             {"button": 1, "pos": (RX + 298, 433)}))
    assert workbench.chart_metric == "progress"
    workbench.cycle_metric(-1)
    assert workbench.chart_metric == "score"
    workbench.cycle_metric(-1)
    assert workbench.chart_metric == "lap"
    workbench.select_algorithm("dqn")
    workbench.cycle_metric(1)
    assert workbench.chart_metric == "reward"
    workbench.select_algorithm("evolution")
    workbench.cycle_metric(1)
    assert workbench.chart_metric == "progress"
    workbench.draw()


def test_pit_wall_pages_and_selects_a_specific_live_car(workbench):
    workbench.tick()
    workbench.toggle_pit_board()
    workbench.draw()
    workbench.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                             {"button": 1, "pos": (775, 598)}))
    assert workbench.board_page == 1
    workbench.draw()
    workbench.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                             {"button": 1, "pos": (136, 267)}))
    assert workbench.trainers["evolution"].index == 20
    assert workbench._world_frame() is workbench.trainers["evolution"].traces[20][-1]
    workbench.draw()
    workbench.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                             {"button": 1, "pos": (650, 225)}))
    assert workbench.board_ranked


def test_decision_sandbox_freezes_and_restores_training(workbench):
    workbench.tick()
    original_steps = workbench.trainer.car.steps
    workbench.open_sandbox()
    assert workbench.paused and workbench.sandbox is not None
    workbench.sandbox.set_input(3)
    workbench.sandbox.set_value(0)
    workbench.draw()
    workbench.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                             {"button": 1, "pos": (770, 469)}))
    assert workbench.sandbox.edited[3] == 1
    workbench.handle_event(pygame.event.Event(pygame.MOUSEMOTION,
                                             {"pos": (142, 469), "rel": (-628, 0), "buttons": (1, 0, 0)}))
    workbench.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP,
                                             {"button": 1, "pos": (142, 469)}))
    workbench.tick()
    assert workbench.trainer.car.steps == original_steps
    assert workbench._sandbox_frame()["observation"][3] == 0
    workbench.draw()
    workbench._close_sandbox()
    assert workbench.sandbox is None and not workbench.paused


def test_recorded_ray_values_use_the_pre_action_car_pose(workbench):
    workbench.trainer.tick()
    frame = workbench._world_frame()
    for i, degrees in enumerate(SENSOR_ANGLES):
        expected = workbench.track.ray_distance(
            frame["decision_x"], frame["decision_y"],
            frame["decision_angle"] + math.radians(degrees)) / 190
        assert frame["observation"][i] == pytest.approx(expected, abs=0.001)


def test_dqn_probe_keeps_the_network_that_made_the_recorded_decision(workbench):
    workbench.select_algorithm("dqn")
    workbench.trainer.tick()
    recorded_outputs = workbench._world_frame()["outputs"]
    workbench.trainer.online.w2 += 0.5  # Mimic a later gradient update.
    workbench.open_sandbox()
    _, baseline, _ = workbench.sandbox.baseline()
    assert baseline == pytest.approx(recorded_outputs, abs=0.001)


def test_wiring_overlay_toggles_and_draws_without_a_trained_frame(workbench):
    assert workbench.wiring is False
    workbench.toggle_wiring()
    assert workbench.wiring is True
    workbench.draw()
    workbench.tick()
    workbench.draw()
    workbench.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}))
    assert workbench.wiring is False
    assert workbench.view == "train"


def test_wiring_key_toggles_and_is_mutually_exclusive_with_other_overlays(workbench):
    workbench.tick()
    workbench.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_n}))
    assert workbench.wiring is True
    workbench.toggle_pit_board()
    assert workbench.wiring is False and workbench.pit_board is True
    workbench.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_n}))
    assert workbench.wiring is True and workbench.pit_board is False
    workbench.open_sandbox()
    assert workbench.wiring is False and workbench.sandbox is not None
    workbench._close_sandbox()


def test_wiring_layout_covers_every_node_and_stays_inside_the_panel(workbench):
    _, _, _, in_ys, hidden_ys, out_ys = workbench._wiring_layout()
    assert len(in_ys) == 12 and len(hidden_ys) == 16 and len(out_ys) == 5
    for ys in (in_ys, hidden_ys, out_ys):
        assert min(ys) >= 240 and max(ys) <= 660


def test_wire_colour_tracks_weight_sign_and_magnitude(workbench):
    strong_positive = workbench._wire_color(1.0, 1.0, True)
    weak_positive = workbench._wire_color(0.05, 1.0, True)
    strong_negative = workbench._wire_color(-1.0, 1.0, True)
    assert strong_positive[1] > strong_positive[0]
    assert strong_negative[0] > strong_negative[1]
    assert strong_positive[1] > weak_positive[1]
    assert workbench._wire_color(1.0, 1.0, False) == (54, 67, 76)


def test_wiring_hover_isolates_a_node_without_touching_the_model(workbench, monkeypatch):
    workbench.tick()
    workbench.toggle_wiring()
    input_x, hidden_x, _, _, hidden_ys, _ = workbench._wiring_layout()
    network = workbench._decision_network()
    before = network.w1.copy()
    drawn = []
    real_line = pygame.draw.line
    monkeypatch.setattr(pygame.draw, "line",
                        lambda *a, **k: (drawn.append(1), real_line(*a, **k))[1])
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (0, 0))
    workbench.draw()
    unfocused = len(drawn)
    drawn.clear()
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (hidden_x, int(hidden_ys[3])))
    workbench.draw()
    focused = len(drawn)
    assert 0 < focused < unfocused
    assert np.array_equal(network.w1, before)
