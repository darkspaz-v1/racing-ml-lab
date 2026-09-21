
import numpy as np
import pygame
import pytest

from racing_lab import app as app_module
from racing_lab.app import App
from racing_lab.scenery import CHECKER, GRASS, _turn_profile, render_track
from racing_lab.simulation import Sensors
from racing_lab.track import Track, apex_circuit, default_track


@pytest.fixture
def workbench(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    lab = App()
    yield lab
    pygame.quit()


@pytest.fixture
def display(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((1, 1))
    yield
    pygame.quit()


def bundled_tracks():
    return [Track.load(path) for path in sorted(app_module.TRACKS.glob("*.json"))]


def test_apex_circuit_is_valid_and_matches_its_bundled_file():
    track = apex_circuit()
    saved = Track.load(app_module.TRACKS / "apex-circuit.json")
    assert saved.definition() == track.definition()
    assert track.name == "Apex Circuit" and len(track.points) > 32


def test_apex_circuit_keeps_separate_parts_of_the_loop_apart():
    """Two stretches of road far apart along the lap must never touch or merge."""
    track = apex_circuit()
    pts = np.array(track.points)
    arc = np.concatenate([[0], np.cumsum(track.lengths)[:-1]])
    gaps = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    along = np.abs(arc[:, None] - arc[None])
    along = np.minimum(along, track.total_length - along)
    far = along > 3.2 * track.width
    assert gaps[far].min() > track.width + 40


def test_turn_profile_follows_the_direction_of_travel():
    # Both bundled loops run clockwise on screen, so right-hand turning dominates.
    for track in (default_track(), apex_circuit()):
        assert _turn_profile(track).sum() > 0
    reverse = Track(list(reversed(default_track().points)), 100)
    assert _turn_profile(reverse).sum() < 0


def test_painted_road_matches_the_physics_boundary(display):
    for track in (default_track(), apex_circuit()):
        pixels = pygame.surfarray.array3d(render_track(track)).swapaxes(0, 1).astype(int)
        grass = np.zeros(track.mask.shape, dtype=bool)
        for colour in GRASS:
            grass |= (np.abs(pixels - np.array(colour)) <= 12).all(axis=2)
        # Nothing the car can drive on is painted as grass.
        assert not (grass & track.mask).any()
        # A point well away from every road is grass or scenery, not asphalt.
        distance_far = pixels[5, 5]
        assert not track.mask[5, 5] and tuple(distance_far) != (57, 61, 67)


def test_chequered_line_is_painted_across_the_start_gate(display):
    track = apex_circuit()
    pixels = pygame.surfarray.array3d(render_track(track)).swapaxes(0, 1)
    x, y, tx, ty, _ = track.checkpoints[0]
    colours = {tuple(pixels[round(y + tx * offset), round(x - ty * offset)])
               for offset in range(-30, 31, 3)}
    assert tuple(CHECKER[0]) in colours and tuple(CHECKER[1]) in colours


def test_the_app_starts_on_the_circuit_and_places_the_garage_off_the_road(workbench):
    assert workbench.track.name == "Apex Circuit"
    for track in bundled_tracks():
        workbench.track = track
        x, y = workbench._garage_origin()
        w, h = workbench.GARAGE_SIZE
        # Dense switchback layouts may not have a panel-sized infield. The
        # search must still choose a mostly clear location rather than hide a
        # large part of the circuit.
        assert track.mask[y:y + h, x:x + w].mean() < 0.15, track.name
    # A track with open infield keeps the garage near its usual place.
    workbench.track = default_track()
    x, y = workbench._garage_origin()
    assert abs(x - workbench.GARAGE_HOME[0]) + abs(y - workbench.GARAGE_HOME[1]) < 80


def test_switchback_park_is_a_dense_hairpin_track():
    track = Track.load(app_module.TRACKS / "switchback-park.json")
    assert track.name == "Switchback Park"
    assert track.width == 70 and len(track.points) >= 60
    horizontal = [abs(b[0] - a[0]) > 30 and abs(b[1] - a[1]) < 12
                  for a, b in track.segments]
    assert sum(horizontal) >= 20


def test_track_button_cycles_through_saved_tracks_and_back(workbench):
    names = [track.name for track in bundled_tracks()]
    seen = [workbench.track.name]
    for _ in names:
        workbench.cycle_track()
        seen.append(workbench.track.name)
    assert sorted(set(seen)) == sorted(names)
    assert seen[-1] == seen[0]
    assert workbench.trainer.generation == 1


def test_track_button_is_clickable_on_the_track_view(workbench):
    workbench.draw()
    workbench.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                             {"button": 1, "pos": (app_module.OX + 520, app_module.OY + 30)}))
    assert workbench.track.name != "Apex Circuit"


def test_results_table_only_compares_runs_on_the_current_track(workbench):
    workbench.settings.max_steps = 40
    workbench.reset_trainers()
    while not workbench.trainer.history:
        workbench.tick()
    workbench.cycle_track()  # records the Apex result, then moves on
    assert workbench.sensor_results[-1]["track"] == "Apex Circuit"
    workbench.toggle_sensor_panel()
    workbench.draw()  # the Apex row is filtered out on the new track without error
    while not workbench.trainer.history:
        workbench.tick()
    workbench.apply_sensors(Sensors(9))
    assert workbench.sensor_results[-1]["track"] == workbench.track.name
    workbench.draw()


def test_art_is_cached_per_track_and_rebuilt_when_it_changes(workbench):
    first = workbench._track_surface()
    assert workbench._track_surface() is first
    workbench.cycle_track()
    assert workbench._track_surface() is not first


def test_every_view_draws_on_the_circuit(workbench):
    workbench.tick()
    for opener in (workbench.start_compare, workbench.start_race, workbench.start_editor,
                   lambda: workbench.select_algorithm("dqn")):
        opener()
        workbench.tick()
        workbench.draw()
