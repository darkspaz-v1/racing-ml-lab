import pygame
import pytest

from racing_lab import app as app_module
from racing_lab.app import App, RX
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
    assert workbench.trainer.best_origin["name"] == "Foundry Loop"
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
    assert workbench.track.name == "Foundry Loop"
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
