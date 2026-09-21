import numpy as np
import pygame
import pytest

from racing_lab import app as app_module
from racing_lab.app import App
from racing_lab.simulation import Car
from racing_lab.track import default_track
from racing_lab.visuals import CAR_COLORS, CAR_STYLES, CarPainter


@pytest.fixture
def workbench(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    lab = App()
    yield lab
    pygame.quit()


def test_every_car_body_has_distinct_rendered_art(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((1, 1))
    painter = CarPainter()
    pixels = [pygame.surfarray.array3d(painter._base(CAR_COLORS[0], False, style))
              for style in CAR_STYLES]
    assert all(not np.array_equal(pixels[i], pixels[j])
               for i in range(len(pixels)) for j in range(i + 1, len(pixels)))
    pygame.quit()


def test_car_garage_changes_player_and_ai_looks(workbench):
    workbench.toggle_car_panel()
    workbench.draw()
    # Formula card for the player's car.
    workbench.handle_event(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (375, 350)}))
    assert workbench.player_style == "formula"
    # Purple paint swatch.
    workbench.draw()
    workbench.handle_event(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (373, 515)}))
    assert workbench.player_color == CAR_COLORS[3]
    # Switch target and choose the rally body for the AI fleet.
    workbench.draw()
    workbench.handle_event(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (360, 237)}))
    workbench.draw()
    workbench.handle_event(pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (555, 350)}))
    assert workbench.car_target == "ai" and workbench.ai_style == "rally"


def test_scaled_race_steering_is_gentler_than_learning_action():
    full = Car(default_track())
    gentle = Car(default_track())
    full.speed = gentle.speed = 3.0
    start = full.angle
    full.step(0, throttle_scale=0)
    gentle.step(0, steering_scale=0.35, throttle_scale=0)
    assert abs(gentle.angle - start) < abs(full.angle - start)
    assert gentle.speed == pytest.approx(full.speed)


def test_keyboard_race_controls_ramp_instead_of_snapping(workbench, monkeypatch):
    class Keys:
        def __getitem__(self, key):
            return key in (pygame.K_UP, pygame.K_LEFT)

    workbench.start_race()
    start_angle = workbench.human.angle
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: Keys())
    workbench.tick()
    assert workbench.race_throttle == pytest.approx(0.075)
    assert workbench.race_steer == pytest.approx(-0.10)
    assert abs(workbench.human.angle - start_angle) < 0.01
    workbench.cycle_race_controls()
    assert workbench.race_controls[workbench.race_control_index][0] == "Balanced"


def test_car_and_control_choices_persist_locally(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setattr(app_module, "PREFERENCES", tmp_path / "preferences.json")
    first = App()
    first.player_style = "formula"
    first.player_color = CAR_COLORS[3]
    first.ai_style = "rally"
    first.race_control_index = 2
    first._save_preferences()
    pygame.quit()
    second = App()
    assert second.player_style == "formula" and second.player_color == CAR_COLORS[3]
    assert second.ai_style == "rally" and second.race_control_index == 2
    pygame.quit()
