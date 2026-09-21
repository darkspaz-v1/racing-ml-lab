"""Pygame learning workbench: simulation, inspection, editor, races, and replays."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import pygame

from .decision_lab import DecisionProbe
from .learning import DQNTrainer, EvolutionTrainer, Settings
from .network import Network
from .simulation import ACTION_NAMES, EXTRA_INPUTS, EXTRA_LABELS, MAX_RAYS, Car, Sensors
from .scenery import render_track
from .track import Track, WORLD_H, WORLD_W, apex_circuit
from .visuals import CAR_COLORS, CAR_STYLES, CarPainter


ROOT = Path(__file__).resolve().parent.parent
TRACKS = ROOT / "data" / "tracks"
MODELS = ROOT / "data" / "models"
REPLAYS = ROOT / "data" / "replays"
PREFERENCES = ROOT / "data" / "preferences.json"
for directory in (TRACKS, MODELS, REPLAYS):
    directory.mkdir(parents=True, exist_ok=True)

W, H = 1500, 900
OX, OY = 20, 83
RX, RW = 940, 540
BG = (10, 17, 28)
PANEL = (19, 29, 43)
PANEL2 = (25, 39, 55)
BORDER = (48, 66, 82)
TEXT = (237, 245, 249)
MUTED = (156, 178, 193)
GREEN = (81, 222, 152)
BLUE = (100, 181, 247)
ORANGE = (255, 181, 89)
RED = (255, 106, 110)
EVOLUTION_COLORS = ((112, 175, 249), (177, 139, 237), (95, 203, 202),
                    (242, 173, 102), (147, 197, 132), (218, 138, 171))


def text(surface, font, message, color, x, y, max_width=None):
    message = str(message)
    if max_width is not None:
        while message and font.size(message)[0] > max_width:
            message = message[:-2] + "…"
    surface.blit(font.render(message, True, color), (int(x), int(y)))


def box(surface, rect, fill=PANEL, border=BORDER, radius=12):
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, 1, border_radius=radius)


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Racing ML Lab")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 17)
        self.small = pygame.font.SysFont("segoeui", 14)
        self.tiny = pygame.font.SysFont("segoeui", 12)
        self.title = pygame.font.SysFont("segoeui", 24, bold=True)
        self.bold = pygame.font.SysFont("segoeui", 16, bold=True)
        self.mono = pygame.font.SysFont("consolas", 14)
        self.car_painter = CarPainter()
        self.player_style = "rally"
        self.player_color = CAR_COLORS[0]
        self.ai_style = "prototype"
        self.ai_color = CAR_COLORS[1]
        self.car_panel = False
        self.car_target = "player"
        self.race_steer = 0.0
        self.race_throttle = 0.0
        self.race_control_index = 0
        self.race_controls = (("Gentle", 0.56, 0.10), ("Balanced", 0.76, 0.15),
                              ("Direct", 1.0, 0.24))
        self._load_preferences()
        self.track = apex_circuit()
        self._track_art = None
        self._garage_slot = None
        self.settings = Settings()
        self.trainers = {"evolution": EvolutionTrainer(self.track, self.settings),
                         "dqn": DQNTrainer(self.track, self.settings)}
        self.algorithm = "evolution"
        self.view = "train"
        self.paused = False
        self.speed_index = 0
        self.speeds = [1, 8, 40, 120]
        self.chart_metric = "score"
        self.pit_board = False
        self.board_page = 0
        self.board_ranked = False
        self.sandbox: DecisionProbe | None = None
        self.wiring = False
        self.sensor_panel = False
        self.sensor_draft: Sensors | None = None
        self.sensor_results: list[dict] = []
        self.sandbox_previous_pause = False
        self.sandbox_dragging = False
        self.buttons: list[tuple[pygame.Rect, object]] = []
        self.message = "Train a driver, inspect its decisions, then race it."
        self.message_time = pygame.time.get_ticks()
        self.editor_points = list(self.track.points)
        self.editor_width = self.track.width
        self.editor_name = self.track.name
        self.editor_gates = list(self.track.gate_fractions)
        self.editor_tool = "points"
        self.editor_selected: int | None = None
        self.dragging = False
        self._editor_preview_key = None
        self._editor_preview = None
        self.replay: dict | None = None
        self.replay_source = "best"
        self.replay_index = 0
        self.human: Car | None = None
        self.opponent: Car | None = None
        self.opponent_network: Network | None = None
        self.race_frame = None

    @property
    def trainer(self):
        return self.trainers[self.algorithm]

    def _load_preferences(self) -> None:
        try:
            saved = json.loads(PREFERENCES.read_text(encoding="utf-8"))
            for target in ("player", "ai"):
                style = saved.get(f"{target}_style")
                color = saved.get(f"{target}_color")
                if style in CAR_STYLES:
                    setattr(self, f"{target}_style", style)
                if (isinstance(color, list) and len(color) == 3 and
                        all(isinstance(channel, int) and 0 <= channel <= 255 for channel in color)):
                    setattr(self, f"{target}_color", tuple(color))
            index = saved.get("race_control_index", 0)
            if isinstance(index, int) and 0 <= index < len(self.race_controls):
                self.race_control_index = index
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    def _save_preferences(self) -> None:
        data = {"player_style": self.player_style, "player_color": self.player_color,
                "ai_style": self.ai_style, "ai_color": self.ai_color,
                "race_control_index": self.race_control_index}
        try:
            PREFERENCES.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    @property
    def sensors(self) -> Sensors:
        """The sensor setup both learners are currently training with."""
        return self.trainers["evolution"].sensors

    def status(self, message: str) -> None:
        self.message = message
        self.message_time = pygame.time.get_ticks()

    def button(self, label, rect, action, accent=False, active=False, disabled=False):
        rect = pygame.Rect(rect)
        hover = rect.collidepoint(pygame.mouse.get_pos()) and not disabled
        fill = (34, 92, 77) if accent else (37, 56, 73) if hover or active else PANEL2
        if disabled:
            fill = (29, 39, 50)
        pygame.draw.rect(self.screen, fill, rect, border_radius=8)
        pygame.draw.rect(self.screen, GREEN if active or accent else BORDER, rect, 1, border_radius=8)
        rendered = self.small.render(label, True, MUTED if disabled else TEXT)
        self.screen.blit(rendered, rendered.get_rect(center=rect.center))
        if not disabled:
            self.buttons.append((rect, action))

    def cycle_metric(self, direction: int) -> None:
        metrics = ["score", "progress", "completion", "crashes", "lap"]
        if self.algorithm == "dqn":
            metrics.append("reward")
        current = self.chart_metric if self.chart_metric in metrics else metrics[0]
        self.chart_metric = metrics[(metrics.index(current) + direction) % len(metrics)]

    def reset_trainers(self):
        self._close_sandbox()
        self.trainers = {"evolution": EvolutionTrainer(self.track, self.settings),
                         "dqn": DQNTrainer(self.track, self.settings)}
        self.board_page = 0
        self.paused = False
        self.status("Both learners reset. Their tracks and seed are identical.")

    def select_algorithm(self, algorithm):
        self._close_sandbox()
        self.pit_board = False
        self.algorithm = algorithm
        self.view = "train"
        self.paused = False
        self.status("Evolution selects and mutates policies." if algorithm == "evolution"
                    else "DQN learns action values from replayed experience.")

    def start_compare(self):
        self._close_sandbox()
        self.pit_board = False
        self.view = "compare"
        self.paused = False
        self.status("Both learners now advance together. Select a model to inspect its decisions.")

    def current_speed(self) -> int:
        if self.view == "compare":
            return (1, 2, 5, 12)[self.speed_index]
        if self.view == "train" and self.algorithm == "evolution":
            return (1, 3, 8, 20)[self.speed_index]
        return self.speeds[self.speed_index]

    def focus_evolution(self, delta: int) -> None:
        population = self.trainers["evolution"]
        self.focus_model((population.index + delta) % len(population.population))

    def focus_model(self, index: int) -> None:
        population = self.trainers["evolution"]
        population.focus(index)
        if self.view == "compare":
            self.algorithm = "evolution"
        self.status(f"Inspecting evolution model E{population.index + 1}.")

    def toggle_pit_board(self) -> None:
        self._close_sandbox()
        self.wiring = False
        self.sensor_panel = False
        self.pit_board = not self.pit_board

    def toggle_car_panel(self) -> None:
        was_open = self.car_panel
        self.car_panel = not self.car_panel
        if was_open:
            self._save_preferences()
        if self.car_panel:
            self._close_sandbox()
            self.pit_board = self.wiring = self.sensor_panel = False
            self.status("Car garage: choose a body and color for your car or the AI fleet.")

    def set_car_style(self, style: str) -> None:
        if style not in CAR_STYLES:
            return
        setattr(self, f"{self.car_target}_style", style)

    def set_car_color(self, color: tuple[int, int, int]) -> None:
        setattr(self, f"{self.car_target}_color", color)

    def cycle_race_controls(self) -> None:
        self.race_control_index = (self.race_control_index + 1) % len(self.race_controls)
        self.status(f"Race steering set to {self.race_controls[self.race_control_index][0].lower()}.")

    def toggle_wiring(self) -> None:
        self._close_sandbox()
        self.pit_board = False
        self.sensor_panel = False
        self.wiring = not self.wiring
        if self.wiring:
            self.status("Network wiring: every input, hidden unit, and action, drawn as nodes and weighted links.")

    def toggle_sensor_panel(self) -> None:
        self._close_sandbox()
        self.pit_board = False
        self.wiring = False
        self.sensor_panel = not self.sensor_panel
        if self.sensor_panel:
            self.sensor_draft = self.sensors
            self.status("Sensor setup: choose rays and inputs, then apply to retrain both learners.")

    def change_draft_rays(self, delta: int) -> None:
        draft = self.sensor_draft or self.sensors
        rays = max(0, min(MAX_RAYS, draft.rays + delta))
        try:
            self.sensor_draft = Sensors(rays, draft.extras)
        except ValueError as exc:
            self.status(str(exc))

    def toggle_draft_input(self, key: str) -> None:
        draft = self.sensor_draft or self.sensors
        extras = tuple(k for k in draft.extras if k != key) if key in draft.extras else draft.extras + (key,)
        try:
            self.sensor_draft = Sensors(draft.rays, extras)
        except ValueError as exc:
            self.status(str(exc))

    def cycle_track(self) -> None:
        """Switch to the next track saved in data/tracks, then retrain on it."""
        files = sorted(TRACKS.glob("*.json"))
        if not files:
            self.status("No saved tracks in data/tracks.")
            return
        current = self.track.definition()
        index = -1
        for i, path in enumerate(files):
            try:
                if Track.load(path).definition() == current:
                    index = i
            except (ValueError, KeyError, OSError, json.JSONDecodeError):
                continue
        for step in range(1, len(files) + 1):
            path = files[(index + step) % len(files)]
            try:
                track = Track.load(path)
            except (ValueError, KeyError, OSError, json.JSONDecodeError):
                continue
            self._record_setup_result()
            self.track = track
            self.reset_trainers()
            self.status(f"Now racing on {track.name}. Both learners restarted on the new circuit.")
            return
        self.status("None of the saved tracks could be loaded.")

    def _track_surface(self) -> pygame.Surface:
        key = json.dumps(self.track.definition())
        if self._track_art is None or self._track_art[0] != key:
            self._track_art = (key, render_track(self.track))
        return self._track_art[1]

    GARAGE_SIZE = (390, 174)
    GARAGE_HOME = (254, 294)   # where the garage is laid out; used when it is clear

    def _garage_origin(self) -> tuple[int, int]:
        """Top-left of the model garage: road-free, and as close to home as possible.

        A summed-area table gives the road pixels under any box in constant time,
        so every candidate position is scored in one vectorised pass.
        """
        key = json.dumps(self.track.definition())
        if self._garage_slot is None or self._garage_slot[0] != key:
            w, h = self.GARAGE_SIZE
            pad, top = 6, 60   # keep a margin from the road; stay below the track labels
            table = np.zeros((WORLD_H + 1, WORLD_W + 1), dtype=np.int64)
            table[1:, 1:] = self.track.mask.cumsum(axis=0).cumsum(axis=1)
            ys = np.arange(top, WORLD_H - h - 8 + 1, 2)
            xs = np.arange(8, WORLD_W - w - 8 + 1, 2)
            y0, y1 = np.clip(ys - pad, 0, WORLD_H), np.clip(ys + h + pad, 0, WORLD_H)
            x0, x1 = np.clip(xs - pad, 0, WORLD_W), np.clip(xs + w + pad, 0, WORLD_W)
            road = (table[np.ix_(y1, x1)] - table[np.ix_(y0, x1)]
                    - table[np.ix_(y1, x0)] + table[np.ix_(y0, x0)])
            home_x, home_y = self.GARAGE_HOME
            distance = np.abs(ys - home_y)[:, None] + np.abs(xs - home_x)[None, :]
            row, column = np.unravel_index(np.argmin(road * 100000 + distance), road.shape)
            self._garage_slot = (key, (int(xs[column]), int(ys[row])))
        return self._garage_slot[1]

    def _record_setup_result(self) -> None:
        result = self._setup_result()
        if result:
            self.sensor_results = (self.sensor_results + [result])[-12:]

    def _setup_result(self) -> dict | None:
        """Summary of how the current sensor setup has trained so far."""
        evolution, dqn = self.trainers["evolution"], self.trainers["dqn"]
        if not evolution.history and not dqn.history:
            return None
        return {"label": self.sensors.short_label(), "track": self.track.name,
                "generations": len(evolution.history),
                "evolution_best": evolution.best_score if math.isfinite(evolution.best_score) else None,
                "lap_rate": max((row["completion"] for row in evolution.history), default=0.0),
                "episodes": len(dqn.history),
                "dqn_best": dqn.best_score if math.isfinite(dqn.best_score) else None,
                "greedy_laps": sum(row["completion"] for row in dqn.evaluation_history)}

    def apply_sensors(self, sensors: Sensors | None = None) -> None:
        sensors = sensors or self.sensor_draft
        if sensors is None or sensors == self.sensors:
            self.status("That sensor setup is already training.")
            return
        self._record_setup_result()
        self.settings.rays, self.settings.inputs = sensors.rays, sensors.extras
        self.reset_trainers()
        self.sensor_draft = sensors
        self.status(f"Now training with {sensors.label()}. Both learners restarted from scratch.")

    def open_sandbox(self) -> None:
        frame = self._world_frame() or {}
        observation = frame.get("observation", [])
        if len(observation) != self.sensors.size:
            self.status("Wait for one driving decision before opening the decision sandbox.")
            return
        network = self._decision_network()
        if network is None:
            self.status("Load a model before inspecting race decisions.")
            return
        self.sandbox = DecisionProbe.capture(network, observation, frame.get("action"), self.sensors)
        self.sandbox_previous_pause = self.paused
        self.paused = True
        self.pit_board = False
        self.wiring = False
        self.sensor_panel = False
        self.status("Decision sandbox: change a sensor to inspect the frozen network. Training is paused.")

    def _close_sandbox(self) -> None:
        if self.sandbox is not None:
            self.paused = self.sandbox_previous_pause
            self.sandbox = None
            self.sandbox_dragging = False

    def _sandbox_frame(self):
        frame = self._world_frame() or {}
        if self.sandbox is None:
            return frame
        hidden, outputs, action = self.sandbox.evaluate()
        return {**frame, "observation": self.sandbox.edited,
                "hidden": hidden, "outputs": outputs, "action": action}

    def _decision_network(self):
        if self.sandbox is not None:
            return self.sandbox.network
        if self.view == "race":
            return self.opponent_network
        if self.algorithm == "dqn" and self.view in ("train", "compare"):
            return self.trainer.last_decision_network
        return self.trainer.current_network

    def start_race(self):
        self.pit_board = False
        self.car_panel = False
        self.view = "race"
        self.paused = False
        self.human = Car(self.track, self.settings.max_steps, self.sensors)
        self.opponent = Car(self.track, self.settings.max_steps, self.sensors)
        side_x, side_y = -math.sin(self.human.angle) * 15, math.cos(self.human.angle) * 15
        self.human.x += side_x
        self.human.y += side_y
        self.opponent.x -= side_x
        self.opponent.y -= side_y
        self.opponent_network = self.trainer.best_network or self.trainer.current_network.copy()
        self.race_frame = None
        self.race_steer = 0.0
        self.race_throttle = 0.0
        self.status("Hold gas, then steer smoothly. Gentle steering is selected by default.")

    def start_editor(self):
        self.pit_board = False
        self.car_panel = False
        self.view = "editor"
        self.paused = True
        self.editor_points = list(self.track.points)
        self.editor_width = self.track.width
        self.editor_name = self.track.name
        self.editor_gates = list(self.track.gate_fractions)
        self.editor_tool = "points"
        self.editor_selected = None
        self.status("Drag points to reshape. Clear to draw a new loop. Point 1 is the start.")

    def apply_editor(self):
        try:
            fractions = sorted(self.editor_gates) if self.editor_gates else None
            self.track = Track(self.editor_points, self.editor_width, self.editor_name, fractions)
        except (ValueError, TypeError) as exc:
            self.status(f"Track needs work: {exc}")
            return
        self.reset_trainers()
        self.view = "train"
        self.status("Track applied. Both training runs restarted on the new course.")

    def set_start(self):
        if self.editor_selected is None or not self.editor_points:
            self.status("Select a centerline point, then choose Set start.")
            return
        i = self.editor_selected
        self.editor_points = self.editor_points[i:] + self.editor_points[:i]
        self.editor_selected = 0
        self.editor_gates = []
        self.status("Start moved to point 1. Gates will be spaced evenly; add custom gates if desired.")

    def change_setting(self, name, direction):
        ranges = {"seed": (1, 9999, 1), "max_steps": (300, 2000, 100),
                  "population": (8, 80, 4), "mutation": (0.04, 0.70, 0.03),
                  "learning_rate": (0.0001, 0.01, 0.0002),
                  "epsilon_decay": (0.90, 0.999, 0.001),
                  "batch_size": (8, 128, 8)}
        low, high, step = ranges[name]
        current = getattr(self.settings, name)
        value = max(low, min(high, current + direction * step))
        if isinstance(current, float):
            value = round(value, 4)
        if value != current:
            setattr(self.settings, name, value)
            self.reset_trainers()
            self.status(f"{name.replace('_', ' ').title()} = {value}. Training restarted for a fair comparison.")

    def _file_dialog(self, kind, folder, extension):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            if kind == "open":
                path = filedialog.askopenfilename(initialdir=folder,
                                                  filetypes=[(extension.upper(), f"*.{extension}")])
            else:
                path = filedialog.asksaveasfilename(initialdir=folder,
                                                    defaultextension=f".{extension}",
                                                    filetypes=[(extension.upper(), f"*.{extension}")])
            root.destroy()
            return Path(path) if path else None
        except Exception as exc:
            self.status(f"File dialog unavailable: {exc}")
            return None

    def save_track(self):
        try:
            track = Track(self.editor_points, self.editor_width, self.editor_name,
                          sorted(self.editor_gates) if self.editor_gates else None)
            path = self._file_dialog("save", TRACKS, "json")
            if path:
                track.save(path)
                self.status(f"Track saved: {path.name}")
        except ValueError as exc:
            self.status(str(exc))

    def open_track(self):
        path = self._file_dialog("open", TRACKS, "json")
        if not path:
            return
        try:
            loaded = Track.load(path)
            self.editor_points = list(loaded.points)
            self.editor_width = loaded.width
            self.editor_name = loaded.name
            self.editor_gates = list(loaded.gate_fractions)
            self.status(f"Loaded {path.name}. Choose Apply to train on it.")
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            self.status(f"Could not load track: {exc}")

    def save_model(self):
        network = self.trainer.best_network
        if network is None:
            self.status("Finish at least one training run before saving a model.")
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = MODELS / f"{self.algorithm}-{stamp}.npz"
        network.save(path, self.algorithm,
                     {"settings": self.settings.as_dict(), "seed": self.settings.seed,
                      "track": self.trainer.best_origin or self.track.definition(),
                      "best_score": self.trainer.best_score if math.isfinite(self.trainer.best_score) else None})
        self.status(f"Model and run settings saved: {path.name}")

    def load_model(self):
        path = self._file_dialog("open", MODELS, "npz")
        if not path:
            return
        try:
            network, mode, metadata = Network.load(path)
            if mode not in self.trainers:
                raise ValueError("Unknown learning mode in model")
            saved = metadata.get("settings", {})
            # Models saved before sensors were configurable used the default setup.
            sensors = Sensors(int(saved.get("rays", 7)), tuple(saved.get("inputs", EXTRA_INPUTS)))
            if network.w1.shape[0] != sensors.size:
                raise ValueError("Model weights do not match its saved sensor setup")
            switched = sensors != self.sensors
            if switched:
                self.apply_sensors(sensors)
            self.algorithm = mode
            self.trainer.best_network = network
            origin = metadata.get("track")
            self.trainer.best_origin = origin
            same_track = (origin is not None and origin.get("points") == [list(p) for p in self.track.points]
                          and origin.get("width") == self.track.width
                          and origin.get("gate_fractions") == self.track.gate_fractions)
            saved_score = metadata.get("best_score")
            self.trainer.best_score = saved_score if same_track and saved_score is not None else -float("inf")
            if mode == "evolution":
                self.trainer.population[0] = network.copy()
            else:
                self.trainer.online = network.copy()
                self.trainer.target = network.copy()
                self.trainer.last_decision_network = network.copy()
            self.view = "train"
            source_track = (origin or {}).get("name", "unknown")
            note = f" Switched sensors to {sensors.label()}." if switched else ""
            self.status(f"Loaded {path.name} from {source_track}. Race it on the current track.{note}")
        except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
            self.status(f"Could not load model: {exc}")

    def save_replay(self):
        replay = (self.replay if self.view == "replay" and self.replay else
                  self.trainer.best_replay if self.replay_source == "best" else self.trainer.last_replay)
        if not replay:
            self.status("Finish a run first; then a replay can be saved.")
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = REPLAYS / f"{self.algorithm}-{stamp}.json"
        data = dict(replay)
        data["track"] = self.track.definition()
        data["settings"] = self.settings.as_dict()
        path.write_text(json.dumps(data), encoding="utf-8")
        self.status(f"Replay saved: {path.name}")

    def open_replay(self):
        self.pit_board = False
        path = self._file_dialog("open", REPLAYS, "json")
        if not path:
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data.get("frames") or "track" not in data:
                raise ValueError("Replay is missing frames or track")
            t = data["track"]
            self.track = Track(t["points"], t["width"], t.get("name", "Replay Track"),
                               t.get("gate_fractions"))
            self.settings = Settings(**data.get("settings", {}))
            self.reset_trainers()
            self.replay = data
            self.replay_index = 0
            self.algorithm = data.get("mode", "evolution")
            self.view = "replay"
            self.paused = True
            self.status(f"Replay loaded: {path.name}. Use play or step through frames.")
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            self.status(f"Could not load replay: {exc}")

    def replay_best(self):
        self.pit_board = False
        self.replay = self.trainer.best_replay or self.trainer.last_replay
        if self.replay is None:
            self.status("Finish a training run to inspect its replay.")
            return
        self.replay_index = 0
        self.replay_source = "best"
        self.view = "replay"
        self.paused = True
        self.status(self.replay["label"] + " — use Play or step frame by frame.")

    def replay_last(self):
        self.pit_board = False
        self.replay = self.trainer.last_replay
        if self.replay is None:
            self.status("Finish a training run to inspect its replay.")
            return
        self.replay_index = 0
        self.replay_source = "last"
        self.view = "replay"
        self.paused = True
        self.status(self.replay["label"] + " — inspect its crash or lap frame by frame.")

    def tick(self):
        if self.paused:
            return
        if self.view in ("train", "compare"):
            for _ in range(self.current_speed()):
                if self.view == "compare":
                    self.trainers["evolution"].tick()
                    self.trainers["dqn"].tick()
                else:
                    self.trainer.tick()
        elif self.view == "race" and self.human and self.opponent:
            keys = pygame.key.get_pressed()
            left = bool(keys[pygame.K_LEFT] or keys[pygame.K_a])
            right = bool(keys[pygame.K_RIGHT] or keys[pygame.K_d])
            gas = bool(keys[pygame.K_UP] or keys[pygame.K_w])
            brake = bool(keys[pygame.K_DOWN] or keys[pygame.K_s])
            _, maximum, response = self.race_controls[self.race_control_index]
            target_steer = (-1.0 if left and not right else 1.0 if right and not left else 0.0)
            steer_rate = response if target_steer else response * 1.5
            self.race_steer += max(-steer_rate, min(steer_rate, target_steer - self.race_steer))
            throttle_target = 1.0 if gas and not brake else 0.0
            throttle_rate = 0.075 if throttle_target else 0.14
            self.race_throttle += max(-throttle_rate,
                                      min(throttle_rate, throttle_target - self.race_throttle))
            if brake:
                action = 4
            elif self.race_steer < -0.02:
                action = 0
            elif self.race_steer > 0.02:
                action = 2
            elif self.race_throttle > 0.02:
                action = 1
            else:
                action = 3
            self.human.step(action, steering_scale=abs(self.race_steer) * maximum,
                            throttle_scale=self.race_throttle)
            decision_pose = (self.opponent.x, self.opponent.y, self.opponent.angle)
            observation = self.opponent.observation()
            hidden, outputs = self.opponent_network.forward(observation)
            ai_action = int(np.argmax(outputs))
            self.opponent.step(ai_action)
            self.race_frame = self.opponent.snapshot(observation, hidden, outputs, ai_action,
                                                     decision_pose)
            if self.human.done and self.opponent.done:
                self.paused = True
        elif self.view == "replay" and self.replay:
            self.replay_index = min(self.replay_index + max(1, self.speeds[self.speed_index] // 8),
                                    len(self.replay["frames"]) - 1)
            if self.replay_index == len(self.replay["frames"]) - 1:
                self.paused = True

    def _world_frame(self):
        if self.view == "replay" and self.replay:
            return self.replay["frames"][self.replay_index]
        if self.view == "race":
            return self.race_frame
        if self.view in ("train", "compare"):
            return self.trainer.frames[-1]
        return None

    def _draw_car(self, surface, x, y, angle, color, label=None, crashed=False, alpha=255,
                  style=None):
        self.car_painter.draw(surface, x, y, angle, color, alpha, crashed,
                              style or self.ai_style)
        if label:
            text(surface, self.small, label, color, x - 12, y - 35)

    def _draw_model_garage(self, world, frame):
        if self.view not in ("train", "compare"):
            return
        x, y = self._garage_origin()
        if (x, y) == self.GARAGE_HOME:
            self._paint_model_garage(world, frame)
            return
        # The garage is laid out at the first slot; shift it where this track has room.
        layer = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
        self._paint_model_garage(layer, frame)
        world.blit(layer, (x - self.GARAGE_HOME[0], y - self.GARAGE_HOME[1]))

    def _paint_model_garage(self, world, frame):
        panel = pygame.Surface((390, 174), pygame.SRCALPHA)
        pygame.draw.rect(panel, (13, 27, 38, 237), panel.get_rect(), border_radius=13)
        pygame.draw.rect(panel, (65, 87, 100, 245), panel.get_rect(), 1, border_radius=13)
        world.blit(panel, (254, 294))
        selected = (f"EVOLUTION MODEL E{self.trainers['evolution'].index + 1}"
                    if self.algorithm == "evolution" else "RL ONLINE MODEL")
        color = ORANGE if self.algorithm == "evolution" else GREEN
        text(world, self.bold, "MODEL GARAGE", TEXT, 272, 306)
        text(world, self.tiny, selected, color, 272, 329)
        self.car_painter.draw_preview(world, (324, 390), color,
                                      bool(frame and frame.get("crashed")), self.ai_style)
        text(world, self.tiny, "REAR", MUTED, 271, 432)
        text(world, self.tiny, "FRONT →", MUTED, 326, 432)
        pygame.draw.line(world, BORDER, (386, 329), (386, 444))
        if self.view == "compare" or self.algorithm == "evolution":
            evolution = self.trainers["evolution"]
            ranked = sorted(range(len(evolution.cars)),
                            key=lambda i: evolution.cars[i].max_progress, reverse=True)
            shown = [evolution.index] + [i for i in ranked if i != evolution.index][:2]
            for row, i in enumerate(shown):
                car = evolution.cars[i]
                label = f"E{i + 1:02d}  {car.max_progress / self.track.total_length * 100:4.0f}%"
                state = "CRASH" if car.crashed else "DONE" if car.done else "DRIVING"
                y = 336 + row * 25
                pygame.draw.circle(world, ORANGE if i == evolution.index else
                                   EVOLUTION_COLORS[i % len(EVOLUTION_COLORS)], (405, y + 6), 5)
                text(world, self.small, label, TEXT, 417, y, 105)
                text(world, self.tiny, state, RED if car.crashed else MUTED, 543, y + 3)
            if self.view == "compare":
                learner = self.trainers["dqn"].car
                pygame.draw.circle(world, GREEN, (405, 417), 5)
                text(world, self.small, f"RL   {learner.max_progress / self.track.total_length * 100:4.0f}%",
                     TEXT, 417, 411)
                text(world, self.tiny, "LEARNING", MUTED, 543, 414)
            else:
                text(world, self.tiny, f"{evolution.active_count} of {len(evolution.cars)} still driving",
                     MUTED, 402, 418)
        else:
            learner = self.trainers["dqn"]
            text(world, self.small, f"Episode {learner.episode} · ε {learner.epsilon:.2f}", TEXT, 402, 342)
            text(world, self.small, "Online: learns each batch", MUTED, 402, 369)
            text(world, self.small, "Target: copied every 400 steps", MUTED, 402, 396)
            text(world, self.small, "Best: greedy evaluated policy", MUTED, 402, 423)
        action = frame.get("action") if frame else None
        action_text = ACTION_NAMES[action] if isinstance(action, int) else "waiting"
        rays, size = self.sensors.rays, self.sensors.size
        text(world, self.tiny,
             f"{rays} RAY{'' if rays == 1 else 'S'}  →  {size} INPUT{'' if size == 1 else 'S'}  →  5 ACTIONS"
             f"     NOW: {action_text}", MUTED, 272, 450, 355)

    def _draw_editor_world(self, surface):
        pts = [(round(x), round(y)) for x, y in self.editor_points]
        if len(pts) >= 2:
            for point in pts:
                pygame.draw.circle(surface, (59, 75, 85), point, int(self.editor_width / 2 + 3))
            pygame.draw.lines(surface, (59, 75, 85), len(pts) >= 5, pts,
                              int(self.editor_width + 6))
            for point in pts:
                pygame.draw.circle(surface, (49, 65, 75), point, int(self.editor_width / 2))
            pygame.draw.lines(surface, (49, 65, 75), len(pts) >= 5, pts,
                              int(self.editor_width))
            pygame.draw.lines(surface, BLUE, len(pts) >= 5, pts, 2)
        for i, (x, y) in enumerate(pts):
            pygame.draw.circle(surface, ORANGE if i == 0 else GREEN if i == self.editor_selected else BLUE,
                               (round(x), round(y)), 8)
            text(surface, self.small, "START" if i == 0 else str(i + 1), TEXT, x + 12, y - 10)
        if len(pts) >= 5:
            preview = self._preview_track()
            if preview:
                for i, (x, y, tx, ty, _) in enumerate(preview.checkpoints):
                    color = ORANGE if i == 0 else (102, 206, 169)
                    p1 = (x - ty * self.editor_width / 2, y + tx * self.editor_width / 2)
                    p2 = (x + ty * self.editor_width / 2, y - tx * self.editor_width / 2)
                    pygame.draw.line(surface, color, p1, p2, 2)
        text(surface, self.small, "POINT TOOL: click to add · drag to move · right click to remove",
             TEXT, 18, 16)
        text(surface, self.small, "GATE TOOL: click road to add · right click gate to remove",
             MUTED, 18, 38)

    def _draw_world(self):
        box(self.screen, pygame.Rect(OX - 1, OY - 1, WORLD_W + 2, WORLD_H + 2),
            (15, 26, 36), BORDER, 8)
        world = pygame.Surface((WORLD_W, WORLD_H))
        if self.view == "editor":
            world.fill((14, 31, 38))
            for gx in range(0, WORLD_W, 40):
                pygame.draw.line(world, (21, 43, 48), (gx, 0), (gx, WORLD_H))
            for gy in range(0, WORLD_H, 40):
                pygame.draw.line(world, (21, 43, 48), (0, gy), (WORLD_W, gy))
            self._draw_editor_world(world)
        else:
            world.blit(self._track_surface(), (0, 0))
            # Checkpoint gates: faint marks across the road, with the next one lit.
            gates = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
            next_gate = self.trainer.car.next_gate if self.view in ("train", "compare") else None
            for i, (x, y, tx, ty, _) in enumerate(self.track.checkpoints):
                if i == 0:
                    continue  # the chequered line is gate 0
                a = (x - ty * self.track.width / 2, y + tx * self.track.width / 2)
                b = (x + ty * self.track.width / 2, y - tx * self.track.width / 2)
                if i == next_gate:
                    pygame.draw.line(gates, (*GREEN, 235), a, b, 3)
                else:
                    pygame.draw.line(gates, (230, 235, 240, 42), a, b, 1)
            world.blit(gates, (0, 0))
            # Painted on the asphalt just before the line, reading along the track.
            x, y, tx, ty, _ = self.track.checkpoints[0]
            degrees = math.degrees(math.atan2(ty, tx))
            if abs(degrees) > 90:
                degrees -= 180 if degrees > 0 else -180
            label = self.tiny.render("START / FINISH", True, (236, 238, 234))
            label.set_alpha(170)
            label = pygame.transform.rotate(label, -degrees)
            world.blit(label, label.get_rect(center=(round(x - tx * 46), round(y - ty * 46))))

            if (self.view == "compare" or
                    self.view == "train" and self.algorithm == "evolution"):
                evolution = self.trainers["evolution"]
                for i, car in enumerate(evolution.cars):
                    if self.algorithm == "evolution" and i == evolution.index:
                        continue  # Draw the inspected driver over the other cars.
                    color = EVOLUTION_COLORS[i % len(EVOLUTION_COLORS)]
                    self._draw_car(world, car.x, car.y, car.angle, color,
                                   crashed=car.crashed, alpha=75 if car.done else 190)
            if self.view == "compare" and self.algorithm != "dqn":
                learner = self.trainers["dqn"].car
                self._draw_car(world, learner.x, learner.y, learner.angle,
                               GREEN, "RL", learner.crashed, alpha=225)

            frame = self._world_frame()
            if frame:
                x, y, angle = frame["x"], frame["y"], frame["angle"]
                sensor_x = frame.get("decision_x", x)
                sensor_y = frame.get("decision_y", y)
                sensor_angle = frame.get("decision_angle", angle)
                obs = frame.get("observation", [])
                for i, degrees in enumerate(self.sensors.angles):
                    ray_length = (obs[i] * 190 if len(obs) > i else
                                  self.track.ray_distance(sensor_x, sensor_y,
                                                          sensor_angle + math.radians(degrees)))
                    theta = sensor_angle + math.radians(degrees)
                    end = (sensor_x + ray_length * math.cos(theta),
                           sensor_y + ray_length * math.sin(theta))
                    pygame.draw.line(world, GREEN if degrees == 0 else (95, 178, 206),
                                     (sensor_x, sensor_y), end, 2)
                    pygame.draw.circle(world, ORANGE, (round(end[0]), round(end[1])), 3)
                selected_color = (self.ai_color if self.view == "race" else
                                  ORANGE if self.algorithm == "evolution" and self.view in ("train", "compare")
                                  else GREEN)
                selected_label = ("AI" if self.view == "race" else
                                  f"E{self.trainers['evolution'].index + 1}" if self.algorithm == "evolution"
                                  else "RL")
                self._draw_car(world, x, y, angle, selected_color, selected_label,
                               frame.get("crashed", False))
                if frame.get("crashed"):
                    text(world, self.bold, "CRASH", RED, x + 18, y - 24)
            if self.view == "race" and self.human:
                self._draw_car(world, self.human.x, self.human.y, self.human.angle,
                               self.player_color, "YOU", self.human.crashed,
                               style=self.player_style)
            if self.view in ("train", "compare"):
                evolution = self.trainers["evolution"]
                label = (f"EVOLUTION: {evolution.active_count}/{len(evolution.cars)} cars driving together"
                         if self.view == "train" and self.algorithm == "evolution" else
                         f"COMPARE: {evolution.active_count} evolution cars + 1 RL learner"
                         if self.view == "compare" else
                         "REINFORCEMENT LEARNING: online policy driving")
                pygame.draw.rect(world, (14, 27, 36), (13, 13, 399, 37), border_radius=7)
                text(world, self.small, label, TEXT, 26, 21)
                self._draw_model_garage(world, frame)
        self.screen.blit(world, (OX, OY))
        if self.view in ("train", "compare"):
            # Drawn after the world blit, which would otherwise cover it.
            self.button(f"Track: {self.track.name}  ›", (OX + 420, OY + 13, 214, 37),
                        self.cycle_track)

    def _draw_pit_board(self):
        if not self.pit_board or self.view not in ("train", "compare"):
            return
        population = self.trainers["evolution"]
        dim = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
        dim.fill((6, 13, 20, 189))
        self.screen.blit(dim, (OX, OY))
        box(self.screen, pygame.Rect(90, 143, 760, 523), (18, 31, 43), (88, 119, 131))
        text(self.screen, self.title, "PIT WALL", TEXT, 111, 164)
        text(self.screen, self.small, "Every card is a separate neural-network driver. Click one to inspect it.",
             MUTED, 111, 199)
        self.button("Close  F", (744, 160, 88, 36), self.toggle_pit_board)
        if self.view == "compare":
            learner = self.trainers["dqn"]
            rl_progress = min(100, learner.car.max_progress / self.track.total_length * 100)
            text(self.screen, self.small,
                 f"RL online model  ·  episode {learner.episode}  ·  {rl_progress:.0f}% progress",
                 GREEN, 111, 221)
            self.button("Inspect RL", (718, 209, 114, 32),
                        lambda: setattr(self, "algorithm", "dqn"),
                        active=self.algorithm == "dqn")
        else:
            text(self.screen, self.small,
                 f"Generation {population.generation}  ·  {population.active_count}/{len(population.cars)} driving",
                 GREEN, 111, 221)
        total = len(population.cars)
        order = list(range(total))
        if self.board_ranked:
            order.sort(key=lambda i: (population.cars[i].laps,
                                      population.cars[i].max_progress), reverse=True)
        self.button("ID view" if self.board_ranked else "Rank view",
                    (594, 209, 116, 32),
                    lambda: setattr(self, "board_ranked", not self.board_ranked),
                    active=self.board_ranked)
        pages = max(1, (total + 19) // 20)
        self.board_page = min(self.board_page, pages - 1)
        for slot in range(20):
            order_index = self.board_page * 20 + slot
            if order_index >= total:
                break
            index = order[order_index]
            car = population.cars[index]
            column, row = slot % 4, slot // 4
            rect = pygame.Rect(111 + column * 181, 248 + row * 63, 173, 56)
            selected = self.algorithm == "evolution" and index == population.index
            hover = rect.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(self.screen, (47, 56, 59) if selected else
                             (35, 54, 66) if hover else (26, 42, 55), rect, border_radius=7)
            pygame.draw.rect(self.screen, ORANGE if selected else BORDER, rect, 1, border_radius=7)
            color = ORANGE if selected else EVOLUTION_COLORS[index % len(EVOLUTION_COLORS)]
            pygame.draw.circle(self.screen, color, (rect.x + 13, rect.y + 15), 5)
            text(self.screen, self.bold, f"E{index + 1:02d}", TEXT, rect.x + 25, rect.y + 4)
            progress = min(100, car.max_progress / self.track.total_length * 100)
            text(self.screen, self.mono, f"{progress:3.0f}%", color, rect.x + 118, rect.y + 5)
            state = "CRASH" if car.crashed else "DONE" if car.done else "DRIVING"
            text(self.screen, self.tiny, f"Lap {car.laps} · {state}",
                 RED if car.crashed else MUTED, rect.x + 12, rect.y + 28)
            pygame.draw.rect(self.screen, (51, 68, 77), (rect.x + 11, rect.bottom - 8, 151, 3))
            if progress:
                pygame.draw.rect(self.screen, color,
                                 (rect.x + 11, rect.bottom - 8, round(151 * progress / 100), 3))
            self.buttons.append((rect, lambda index=index: self.focus_model(index)))
        text(self.screen, self.small, f"Page {self.board_page + 1} / {pages}", TEXT, 390, 587)
        self.button("‹ Previous", (112, 579, 111, 38),
                    lambda: setattr(self, "board_page", max(0, self.board_page - 1)),
                    disabled=self.board_page == 0)
        self.button("Next ›", (721, 579, 111, 38),
                    lambda: setattr(self, "board_page", min(pages - 1, self.board_page + 1)),
                    disabled=self.board_page >= pages - 1)
        text(self.screen, self.tiny,
             "Progress is based on ordered gates. Crashed cars stay visible until the next generation.",
             MUTED, 111, 631)

    def _set_sandbox_slider(self, mouse_x: int) -> None:
        if self.sandbox is None:
            return
        low, high = self.sandbox.bounds()
        fraction = max(0.0, min(1.0, (mouse_x - 142) / 628))
        self.sandbox.set_value(low + fraction * (high - low))

    def _draw_sandbox(self):
        if self.sandbox is None:
            return
        probe = self.sandbox
        dim = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
        dim.fill((6, 13, 20, 199))
        self.screen.blit(dim, (OX, OY))
        footer_dim = pygame.Surface((WORLD_W, 146), pygame.SRCALPHA)
        footer_dim.fill((6, 13, 20, 175))
        self.screen.blit(footer_dim, (OX, 739))
        box(self.screen, pygame.Rect(100, 151, 740, 526), (18, 31, 43), (88, 119, 131))
        text(self.screen, self.title, "DECISION SANDBOX", TEXT, 122, 171)
        self.button("Close  Esc", (722, 168, 101, 36), self._close_sandbox)
        text(self.screen, self.small,
             "Change one input. A copy of that decision's network recalculates without driving or training.",
             MUTED, 122, 204, 690)
        text(self.screen, self.tiny, "CHOOSE AN INPUT", BLUE, 122, 225)
        names = probe.sensors.names
        columns = 3 if len(names) <= 12 else 4
        rows = math.ceil(len(names) / columns)
        column_width = 696 // columns
        row_height = min(38, 156 / rows)
        for index, name in enumerate(names):
            column, row = index % columns, index // columns
            value = probe.edited[index]
            self.button(f"{name}   {value:+.2f}",
                        (122 + column * column_width, round(245 + row * row_height),
                         column_width - 16, round(row_height) - 6),
                        lambda index=index: probe.set_input(index),
                        active=probe.selected_input == index)
        low, high = probe.bounds()
        index = probe.selected_input
        text(self.screen, self.bold, f"Adjust {names[index]}", TEXT, 122, 405)
        self.button("Reset inputs", (688, 405, 135, 34), probe.reset)
        text(self.screen, self.small, f"Original {probe.original[index]:+.2f}", MUTED, 122, 436)
        text(self.screen, self.small, f"What if {probe.edited[index]:+.2f}", ORANGE, 651, 436)
        pygame.draw.line(self.screen, BORDER, (142, 469), (770, 469), 9)
        fraction = (probe.edited[index] - low) / (high - low)
        thumb_x = round(142 + fraction * 628)
        pygame.draw.line(self.screen, ORANGE, (142, 469), (thumb_x, 469), 9)
        pygame.draw.circle(self.screen, TEXT, (thumb_x, 469), 11)
        text(self.screen, self.tiny, f"{low:+.0f}", MUTED, 123, 484)
        text(self.screen, self.tiny, f"{high:+.0f}", MUTED, 772, 484)
        _, baseline, greedy = probe.baseline()
        _, edited, prediction = probe.evaluate()
        actual = (ACTION_NAMES[probe.actual_action] if isinstance(probe.actual_action, int) else "waiting")
        text(self.screen, self.tiny,
             f"Recorded action: {actual}  ·  Baseline highest: {ACTION_NAMES[greedy]}  ·  What-if highest: {ACTION_NAMES[prediction]}",
             TEXT, 122, 504, 690)
        text(self.screen, self.tiny, "ACTION", BLUE, 122, 530)
        text(self.screen, self.tiny, "BASELINE", BLUE, 525, 530)
        text(self.screen, self.tiny, "WHAT IF", ORANGE, 686, 530)
        for i, name in enumerate(ACTION_NAMES):
            y = 550 + i * 19
            if prediction == i:
                pygame.draw.rect(self.screen, (65, 57, 43), (120, y - 1, 702, 18), border_radius=3)
            text(self.screen, self.small, name, ORANGE if prediction == i else TEXT, 126, y - 1)
            text(self.screen, self.mono, f"{baseline[i]:+.2f}", MUTED, 524, y - 2)
            text(self.screen, self.mono, f"{edited[i]:+.2f}", ORANGE if prediction == i else TEXT,
                 684, y - 2)
        text(self.screen, self.tiny,
             "Baseline uses the recorded decision's network. What-if values are hypothetical; no car step occurs.",
             MUTED, 122, 650, 690)

    def _wiring_layout(self):
        """Node centres for the wiring diagram: every input, 16 hidden units, 5 actions."""
        input_x, hidden_x, output_x = 268, 508, 664
        count = self.sensors.size
        spacing = min(30, 330 / max(1, count - 1))
        top = 417 - spacing * (count - 1) / 2
        in_ys = [top + i * spacing for i in range(count)]
        hidden_ys = [250 + i * 22.4 for i in range(16)]
        out_ys = [272 + i * 72 for i in range(5)]
        return input_x, hidden_x, output_x, in_ys, hidden_ys, out_ys

    @staticmethod
    def _wire_color(weight, scale, live):
        """Fade each link from the panel colour toward green (+) or red (-) by |weight|."""
        if not live:
            return (54, 67, 76)
        strength = min(1.0, abs(weight) / scale) if scale else 0.0
        target = (41, 132, 108) if weight >= 0 else (138, 64, 74)
        base = (18, 31, 43)
        mix = 0.18 + 0.82 * strength
        return tuple(int(b + (t - b) * mix) for b, t in zip(base, target))

    def _draw_wiring(self):
        if not self.wiring:
            return
        network = self._decision_network()
        dim = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
        dim.fill((6, 13, 20, 199))
        self.screen.blit(dim, (OX, OY))
        footer_dim = pygame.Surface((WORLD_W, 146), pygame.SRCALPHA)
        footer_dim.fill((6, 13, 20, 175))
        self.screen.blit(footer_dim, (OX, 739))
        box(self.screen, pygame.Rect(100, 151, 740, 526), (18, 31, 43), (88, 119, 131))
        text(self.screen, self.title, "NETWORK WIRING", TEXT, 122, 171)
        self.button("Close  N", (722, 168, 101, 36), self.toggle_wiring)
        if network is None:
            text(self.screen, self.small, "Load or train a model to see its wiring.", MUTED, 122, 210)
            return
        text(self.screen, self.small,
             f"Every input, hidden unit, and action, with all {network.w1.size + network.w2.size} "
             "weighted connections.",
             MUTED, 122, 204, 580)

        frame = self._world_frame() or {}
        obs = np.asarray(frame.get("observation", []), dtype=float)
        hidden = np.asarray(frame.get("hidden", []), dtype=float)
        outputs = np.asarray(frame.get("outputs", []), dtype=float)
        action = frame.get("action")
        # Replays store node values but not the weight matrices of the moment.
        live = self.view != "replay"
        input_x, hidden_x, output_x, in_ys, hidden_ys, out_ys = self._wiring_layout()

        mouse = pygame.mouse.get_pos()
        hover_input = next((i for i, y in enumerate(in_ys)
                            if math.dist(mouse, (input_x, y)) <= 9), None)
        hover_hidden = next((i for i, y in enumerate(hidden_ys)
                             if math.dist(mouse, (hidden_x, y)) <= 9), None)
        hover_output = next((i for i, y in enumerate(out_ys)
                             if math.dist(mouse, (output_x, y)) <= 11), None)
        focused = hover_input is not None or hover_hidden is not None or hover_output is not None

        w1_scale = float(np.abs(network.w1).max()) or 1.0
        w2_scale = float(np.abs(network.w2).max()) or 1.0
        for i, y1 in enumerate(in_ys):
            for j, y2 in enumerate(hidden_ys):
                lit = hover_input == i or hover_hidden == j
                if focused and not lit:
                    continue
                weight = network.w1[i, j]
                pygame.draw.line(self.screen, self._wire_color(weight, w1_scale, live),
                                 (input_x, int(y1)), (hidden_x, int(y2)), 2 if lit else 1)
        for i, y1 in enumerate(hidden_ys):
            for j, y2 in enumerate(out_ys):
                lit = hover_hidden == i or hover_output == j
                if focused and not lit:
                    continue
                weight = network.w2[i, j]
                pygame.draw.line(self.screen, self._wire_color(weight, w2_scale, live),
                                 (hidden_x, int(y1)), (output_x, int(y2)), 2 if lit else 1)

        names = self.sensors.names
        for i, y in enumerate(in_ys):
            value = float(obs[i]) if len(obs) == len(names) else 0.0
            text(self.screen, self.tiny, names[i], MUTED, 124, int(y) - 7, 76)
            text(self.screen, self.mono, f"{value:+.2f}",
                 TEXT if hover_input == i else MUTED, 246 - 44, int(y) - 8)
            pygame.draw.circle(self.screen, (76, int(110 + min(1, abs(value)) * 125), 148),
                               (input_x, int(y)), 6)
            if hover_input == i:
                pygame.draw.circle(self.screen, TEXT, (input_x, int(y)), 9, 1)
        for i, y in enumerate(hidden_ys):
            value = float(hidden[i]) if len(hidden) == 16 else 0.0
            intensity = int(75 + min(1, abs(value)) * 160)
            pygame.draw.circle(self.screen, (intensity, 116, 78) if value < 0 else (76, intensity, 130),
                               (hidden_x, int(y)), 5)
            if hover_hidden == i:
                pygame.draw.circle(self.screen, TEXT, (hidden_x, int(y)), 8, 1)
        for i, y in enumerate(out_ys):
            selected = action == i
            pygame.draw.circle(self.screen, ORANGE if selected else BLUE,
                               (output_x, int(y)), 9 if selected else 7)
            if hover_output == i:
                pygame.draw.circle(self.screen, TEXT, (output_x, int(y)), 12, 1)
            value = f"  {outputs[i]:+.2f}" if len(outputs) == 5 else ""
            text(self.screen, self.small, ACTION_NAMES[i] + value,
                 ORANGE if selected else TEXT, output_x + 18, int(y) - 8, 152)

        text(self.screen, self.tiny, "INPUTS", BLUE, 124, 228)
        text(self.screen, self.tiny, "HIDDEN", GREEN, hidden_x - 22, 228)
        text(self.screen, self.tiny, "ACTIONS", ORANGE, output_x - 20, 228)
        if not live:
            note = "Replay stores node values and the action taken; the weights of that moment are not saved."
        elif hover_input is not None:
            column = network.w1[hover_input]
            note = (f"{names[hover_input]} → 16 hidden units  ·  strongest {column.max():+.2f} / "
                    f"weakest {column.min():+.2f}")
        elif hover_hidden is not None:
            incoming = network.w1[:, hover_hidden]
            outgoing = network.w2[hover_hidden]
            strongest_in = int(np.argmax(np.abs(incoming)))
            strongest_out = int(np.argmax(np.abs(outgoing)))
            note = (f"H{hover_hidden + 1}  ·  strongest input {names[strongest_in]} "
                    f"{incoming[strongest_in]:+.2f}  →  strongest action {ACTION_NAMES[strongest_out]} "
                    f"{outgoing[strongest_out]:+.2f}")
        elif hover_output is not None:
            column = network.w2[:, hover_output]
            note = (f"{ACTION_NAMES[hover_output]} ← 16 hidden units  ·  strongest {column.max():+.2f} / "
                    f"weakest {column.min():+.2f}")
        else:
            note = "Green links add, red links subtract; brighter means a larger weight. Hover any node to isolate its connections."
        text(self.screen, self.tiny, note, MUTED, 122, 614, 700)
        text(self.screen, self.tiny,
             "Node fill shows the current value; line colour shows the learned weight. Weights change as training continues.",
             MUTED, 122, 638, 700)

    SENSOR_PRESETS = (("Original", Sensors()),
                      ("Rays + speed", Sensors(7, ("speed",))),
                      ("Rays only", Sensors(7, ())),
                      ("No rays", Sensors(0)),
                      ("15 rays", Sensors(15)))
    INPUT_HINTS = {"speed": "How fast the car is moving.",
                   "goal": "Bearing to the next gate. Route hint.",
                   "lane": "Distance from the centre line. Route hint.",
                   "route": "Angle to the road direction. Route hint."}

    def _draw_sensor_panel(self):
        if not self.sensor_panel or self.view not in ("train", "compare"):
            return
        draft = self.sensor_draft or self.sensors
        dim = pygame.Surface((WORLD_W, WORLD_H), pygame.SRCALPHA)
        dim.fill((6, 13, 20, 199))
        self.screen.blit(dim, (OX, OY))
        footer_dim = pygame.Surface((WORLD_W, 146), pygame.SRCALPHA)
        footer_dim.fill((6, 13, 20, 175))
        self.screen.blit(footer_dim, (OX, 739))
        box(self.screen, pygame.Rect(100, 151, 740, 526), (18, 31, 43), (88, 119, 131))
        text(self.screen, self.title, "SENSOR SETUP", TEXT, 122, 171)
        self.button("Close  R", (722, 168, 101, 36), self.toggle_sensor_panel)
        text(self.screen, self.small,
             "Choose what the car can sense. Applying restarts both learners with a matching network.",
             MUTED, 122, 204, 690)

        # Rays: stepper plus a fan preview drawn at the real sensor angles.
        text(self.screen, self.tiny, "RAYS", BLUE, 122, 232)
        self.button("−", (122, 252, 40, 36), lambda: self.change_draft_rays(-1),
                    disabled=draft.rays == 0)
        pygame.draw.rect(self.screen, PANEL2, (168, 252, 88, 36), border_radius=8)
        rendered = self.title.render(str(draft.rays), True, GREEN)
        self.screen.blit(rendered, rendered.get_rect(center=(212, 270)))
        self.button("+", (262, 252, 40, 36), lambda: self.change_draft_rays(1),
                    disabled=draft.rays == MAX_RAYS)
        text(self.screen, self.tiny, f"0 to {MAX_RAYS}, spread across 180°", MUTED, 312, 263)
        preview = pygame.Rect(122, 300, 330, 164)
        pygame.draw.rect(self.screen, (13, 24, 34), preview, border_radius=8)
        cx, cy = preview.centerx, preview.bottom - 26
        for degrees in draft.angles:
            theta = -math.pi / 2 + math.radians(degrees)
            end = (cx + 118 * math.cos(theta), cy + 118 * math.sin(theta))
            pygame.draw.line(self.screen, GREEN if degrees == 0 else (95, 178, 206), (cx, cy), end, 2)
            pygame.draw.circle(self.screen, ORANGE, (round(end[0]), round(end[1])), 3)
        self._draw_car(self.screen, cx, cy, -math.pi / 2, ORANGE)
        if not draft.rays:
            text(self.screen, self.small, "No rays: the car cannot see walls", MUTED,
                 preview.x + 60, preview.y + 60)

        # Extra inputs: one toggle each.
        text(self.screen, self.tiny, "EXTRA INPUTS", BLUE, 480, 232)
        for i, key in enumerate(EXTRA_INPUTS):
            y = 252 + i * 53
            on = key in draft.extras
            count = len(EXTRA_INPUTS[key][0])
            text(self.screen, self.bold, EXTRA_LABELS[key], TEXT if on else MUTED, 480, y)
            text(self.screen, self.tiny,
                 f"{count} input{'s' if count > 1 else ''} · {self.INPUT_HINTS[key]}",
                 MUTED, 480, y + 21, 262)
            self.button("On" if on else "Off", (752, y + 3, 68, 32),
                        lambda key=key: self.toggle_draft_input(key), active=on)

        text(self.screen, self.tiny, "PRESETS", BLUE, 122, 472)
        for i, (label, preset) in enumerate(self.SENSOR_PRESETS):
            self.button(label, (122 + i * 140, 488, 132, 30),
                        lambda preset=preset: setattr(self, "sensor_draft", preset),
                        active=draft == preset)

        changed = draft != self.sensors
        weights = draft.size * 16 + 16 * 5
        text(self.screen, self.small, draft.label(), ORANGE if changed else TEXT, 122, 530, 500)
        text(self.screen, self.tiny,
             f"{draft.size} → 16 hidden → 5 actions · {weights} weights"
             + ("  ·  not applied yet" if changed else "  ·  training now"),
             MUTED, 122, 551, 500)
        self.button("Apply & restart", (640, 528, 180, 38), self.apply_sensors,
                    accent=changed, disabled=not changed)

        # Results so far, per setup, so a restart does not lose earlier numbers.
        text(self.screen, self.tiny, f"RESULTS BY SETUP · {self.track.name.upper()}", BLUE, 122, 578, 300)
        text(self.screen, self.tiny, "EVOLUTION  gens · best · lap rate", MUTED, 438, 578)
        text(self.screen, self.tiny, "DQN  episodes · best greedy", MUTED, 650, 578)
        current = self._setup_result()
        entries = [(row, False) for row in self.sensor_results
                   if row["track"] == self.track.name][-3:]
        if current:
            entries.append((current, True))
        if not entries:
            text(self.screen, self.tiny, "Train for a generation or episode; results appear here.",
                 MUTED, 122, 598)
        for i, (row, live) in enumerate(entries):
            y = 598 + i * 18
            color = TEXT if live else MUTED
            best = "–" if row["evolution_best"] is None else f"{row['evolution_best']:.0f}"
            dqn_best = "–" if row["dqn_best"] is None else f"{row['dqn_best']:.0f}"
            if live:
                pygame.draw.circle(self.screen, GREEN, (126, y + 8), 3)
            text(self.screen, self.tiny, row["label"] + ("  · now" if live else ""), color, 134, y, 296)
            text(self.screen, self.tiny,
                 f"{row['generations']} · {best} · {row['lap_rate'] * 100:.0f}%", color, 438, y, 200)
            text(self.screen, self.tiny, f"{row['episodes']} · {dqn_best}", color, 650, y, 170)
        text(self.screen, self.tiny,
             "Compare setups after similar training time; more generations alone raise the scores.",
             MUTED, 122, 657, 700)

    def _draw_header(self):
        text(self.screen, self.title, "RACING ML LAB", TEXT, 22, 17)
        text(self.screen, self.small, "LEARN THE DRIVER, INSPECT THE DECISION", MUTED, 23, 47)
        self.button("Reinforcement learning", (329, 18, 188, 42),
                    lambda: self.select_algorithm("dqn"), active=self.view == "train" and self.algorithm == "dqn")
        self.button("Evolution over generations", (525, 18, 218, 42),
                    lambda: self.select_algorithm("evolution"), active=self.view == "train" and self.algorithm == "evolution")
        self.button("Compare models", (751, 18, 124, 42), self.start_compare,
                    active=self.view == "compare")
        self.button("Race AI", (883, 18, 82, 42), self.start_race,
                    active=self.view == "race")
        self.button("Cars", (973, 18, 76, 42), self.toggle_car_panel,
                    active=self.car_panel)
        self.button("Track editor", (1057, 18, 102, 42), self.start_editor,
                    active=self.view == "editor")
        self.button("Replay", (1167, 18, 74, 42), self.replay_best,
                    active=self.view == "replay")
        self.button("How it works", (1249, 18, 102, 42),
                    lambda: setattr(self, "view", "guide"), active=self.view == "guide")
        text(self.screen, self.tiny, "LOCAL ONLY", GREEN, 1381, 30)

    def _draw_car_panel(self):
        if not self.car_panel:
            return
        dim = pygame.Surface((WORLD_W, WORLD_H + 146), pygame.SRCALPHA)
        dim.fill((6, 13, 20, 202))
        self.screen.blit(dim, (OX, OY))
        box(self.screen, pygame.Rect(86, 132, 768, 551), (18, 31, 43), (88, 119, 131))
        text(self.screen, self.title, "CAR GARAGE", TEXT, 110, 153)
        text(self.screen, self.small,
             "Pick the silhouette and paint. AI body choices also style every learning car.",
             MUTED, 110, 187, 610)
        self.button("Close  C", (742, 151, 91, 36), self.toggle_car_panel)
        self.button("My race car", (110, 218, 157, 39),
                    lambda: setattr(self, "car_target", "player"),
                    active=self.car_target == "player")
        self.button("AI + learning fleet", (277, 218, 182, 39),
                    lambda: setattr(self, "car_target", "ai"),
                    active=self.car_target == "ai")
        target_color = self.player_color if self.car_target == "player" else self.ai_color
        target_style = self.player_style if self.car_target == "player" else self.ai_style
        target_label = "YOU" if self.car_target == "player" else "AI"
        text(self.screen, self.small,
             f"Editing {target_label}  ·  {CAR_STYLES[target_style][0]}  ·  changes apply immediately",
             GREEN, 481, 228, 340)
        text(self.screen, self.tiny, "BODY STYLE", BLUE, 110, 276)
        for i, (style, (name, description)) in enumerate(CAR_STYLES.items()):
            x = 110 + i * 181
            rect = pygame.Rect(x, 298, 169, 145)
            active = style == target_style
            pygame.draw.rect(self.screen, (40, 58, 69) if active else PANEL2,
                             rect, border_radius=9)
            pygame.draw.rect(self.screen, GREEN if active else BORDER, rect, 2 if active else 1,
                             border_radius=9)
            self.car_painter.draw_preview(self.screen, (x + 84, 347), target_color,
                                          style=style)
            text(self.screen, self.bold, name, TEXT, x + 13, 393)
            text(self.screen, self.tiny, description, MUTED, x + 13, 416, 145)
            self.buttons.append((rect, lambda style=style: self.set_car_style(style)))
        text(self.screen, self.tiny, "PAINT COLOR", BLUE, 110, 468)
        for i, color in enumerate(CAR_COLORS):
            x = 119 + i * 76
            rect = pygame.Rect(x, 493, 52, 45)
            pygame.draw.rect(self.screen, (31, 46, 58), rect, border_radius=8)
            pygame.draw.circle(self.screen, color, rect.center, 14)
            if color == target_color:
                pygame.draw.rect(self.screen, TEXT, rect, 2, border_radius=8)
                pygame.draw.circle(self.screen, BG, rect.center, 4)
            self.buttons.append((rect, lambda color=color: self.set_car_color(color)))
        pygame.draw.line(self.screen, BORDER, (110, 558), (830, 558))
        self.car_painter.draw_preview(self.screen, (215, 612), self.player_color,
                                      style=self.player_style)
        self.car_painter.draw_preview(self.screen, (510, 612), self.ai_color,
                                      style=self.ai_style)
        text(self.screen, self.bold, "YOUR CAR", self.player_color, 282, 584)
        text(self.screen, self.small, CAR_STYLES[self.player_style][0], MUTED, 282, 610)
        text(self.screen, self.bold, "AI FLEET", self.ai_color, 575, 584)
        text(self.screen, self.small, CAR_STYLES[self.ai_style][0], MUTED, 575, 610)
        text(self.screen, self.tiny,
             "Appearance does not affect physics, sensors, scores, or learning.", MUTED, 110, 657)

    def _draw_inspector(self):
        rect = pygame.Rect(RX, 83, RW, 273)
        box(self.screen, rect)
        text(self.screen, self.bold, "LIVE DECISION", TEXT, RX + 18, 95)
        if self.view == "compare":
            self.button("Evolution", (RX + 305, 91, 94, 31),
                        lambda: setattr(self, "algorithm", "evolution"),
                        active=self.algorithm == "evolution")
            self.button("RL driver", (RX + 407, 91, 115, 31),
                        lambda: setattr(self, "algorithm", "dqn"),
                        active=self.algorithm == "dqn")
        subtitle = ("SANDBOX · hypothetical input, frozen network"
                    if self.sandbox is not None else
                    f"Watching car E{self.trainers['evolution'].index + 1} · largest policy score acts"
                    if self.algorithm == "evolution" and self.view in ("train", "compare") else
                    "DQN · Q values estimate future reward" if self.algorithm == "dqn" else
                    "Policy scores · largest value acts")
        text(self.screen, self.small, subtitle, MUTED, RX + 18, 119, 500)
        frame = self._sandbox_frame()
        obs = np.asarray(frame.get("observation", []), dtype=float)
        hidden = np.asarray(frame.get("hidden", []), dtype=float)
        outputs = np.asarray(frame.get("outputs", []), dtype=float)
        action = frame.get("action")
        # Three stages expose actual values without burying them under 272 weight lines.
        for x in (RX + 182, RX + 306):
            pygame.draw.line(self.screen, BORDER, (x, 150), (x, 319))
        size = self.sensors.size
        text(self.screen, self.tiny, f"01  SENSE · {size} INPUT{'' if size == 1 else 'S'}", BLUE, RX + 18, 145)
        text(self.screen, self.tiny, "02  THINK · 16 UNITS", GREEN, RX + 194, 145)
        text(self.screen, self.tiny, "03  ACT · 5 VALUES", ORANGE, RX + 318, 145)
        self._draw_sense_column(obs)
        hovered = None
        for i in range(16):
            x = RX + 206 + (i % 4) * 24
            y = 184 + (i // 4) * 34
            value = float(hidden[i]) if len(hidden) == 16 else 0.0
            strength = min(1, abs(value))
            color = (int(45 + 39 * strength), int(89 + 150 * strength),
                     int(88 + 48 * strength)) if value >= 0 else (
                     int(92 + 135 * strength), int(70 + 55 * strength), 75)
            pygame.draw.circle(self.screen, color, (x, y), 10)
            if math.dist(pygame.mouse.get_pos(), (x, y)) <= 11:
                hovered = i
                pygame.draw.circle(self.screen, TEXT, (x, y), 12, 1)
            text(self.screen, self.tiny, str(i + 1), BG, x - (6 if i >= 9 else 3), y - 7)
        text(self.screen, self.tiny, "Green +   Red −", MUTED, RX + 194, 315)
        scale = max(1.0, *(abs(float(v)) for v in outputs)) if len(outputs) == 5 else 1.0
        for i, name in enumerate(ACTION_NAMES):
            y = 166 + i * 31
            selected = action == i
            if selected:
                pygame.draw.rect(self.screen, (65, 57, 43),
                                 (RX + 314, y - 3, 208, 30), border_radius=5)
            value = float(outputs[i]) if len(outputs) == 5 else 0.0
            text(self.screen, self.tiny, name, ORANGE if selected else TEXT,
                 RX + 320, y, 125)
            text(self.screen, self.mono, f"{value:+.2f}", ORANGE if selected else MUTED,
                 RX + 463, y - 2)
            pygame.draw.rect(self.screen, (43, 57, 67),
                             (RX + 320, y + 19, 190, 4), border_radius=2)
            pygame.draw.rect(self.screen, ORANGE if selected else BLUE if value >= 0 else RED,
                             (RX + 320, y + 19, max(2, round(abs(value) / scale * 190)), 4),
                             border_radius=2)
        footer = pygame.Rect(RX + 14, 329, RW - 28, 20)
        pygame.draw.rect(self.screen, (27, 43, 55), footer, border_radius=4)
        if (hovered is not None and self.view != "replay" and len(obs) == self.sensors.size
                and len(hidden) == 16):
            network = self._decision_network()
            contributions = obs * network.w1[:, hovered]
            strongest_input = int(np.argmax(np.abs(contributions)))
            outgoing = hidden[hovered] * network.w2[hovered]
            strongest_output = int(np.argmax(np.abs(outgoing)))
            note = (f"H{hovered + 1}: {self.sensors.names[strongest_input]} {contributions[strongest_input]:+.2f}"
                    f"  →  {ACTION_NAMES[strongest_output]} {outgoing[strongest_output]:+.2f}")
        elif self.sandbox is not None:
            note = "Hypothetical model output only · car position and learning do not change."
        elif self.view == "replay":
            note = "Replay shows recorded activity and action. Historical weights are not saved."
        elif self.algorithm == "dqn" and self.view in ("train", "compare"):
            note = ("EXPLORING · random action chosen" if self.trainer.explored else
                    "GREEDY · highest Q value chosen") + "    ·    Hover a hidden unit for its strongest links"
        else:
            note = "Orange is the action taken. Hover a hidden unit for its strongest links."
        text(self.screen, self.tiny, note, MUTED, footer.x + 6, footer.y + 3, footer.width - 12)

    def _sense_row(self, name, value, y):
        text(self.screen, self.tiny, name, MUTED, RX + 18, y - 2, 76)
        pygame.draw.rect(self.screen, (39, 55, 69), (RX + 96, y + 1, 46, 5), border_radius=2)
        width = round(min(1, abs(value)) * 46)
        if width:
            pygame.draw.rect(self.screen, BLUE if value >= 0 else RED,
                             (RX + 96, y + 1, width, 5), border_radius=2)
        text(self.screen, self.tiny, f"{value:+.2f}", TEXT, RX + 145, y - 3)

    def _draw_sense_column(self, obs):
        sensors = self.sensors
        values = [float(v) for v in obs] if len(obs) == sensors.size else [0.0] * sensors.size
        if sensors.size <= 12:
            for i, name in enumerate(sensors.names):
                self._sense_row(name, values[i], 165 + i * 13)
            return
        # Too many rows to list: draw the rays as a fan of bars, left to right,
        # and keep the extra inputs as labelled rows underneath.
        rays = sensors.rays
        chart = pygame.Rect(RX + 18, 162, 158, 52)
        pygame.draw.rect(self.screen, (27, 43, 55), chart, border_radius=4)
        slot = chart.width / rays
        bar = max(3, int(slot) - 3)
        for i, degrees in enumerate(sensors.angles):
            height = round(min(1, max(0, values[i])) * (chart.height - 6))
            x = round(chart.x + i * slot + (slot - bar) / 2)
            pygame.draw.rect(self.screen, GREEN if degrees == 0 else BLUE,
                             (x, chart.bottom - 3 - height, bar, height), border_radius=1)
        text(self.screen, self.tiny, f"{rays} RAYS  ·  left → right", MUTED, RX + 18, 216, 160)
        for row, i in enumerate(range(rays, sensors.size)):
            self._sense_row(sensors.names[i], values[i], 239 + row * 13)

    def _metric_values(self):
        metric = "score" if self.algorithm == "evolution" and self.chart_metric == "reward" else self.chart_metric
        rows = (self.trainer.evaluation_history[-45:]
                if self.algorithm == "dqn" and metric == "score"
                else self.trainer.history[-45:])
        if metric == "score":
            values = [row["best"] for row in rows] if self.algorithm == "evolution" else [row["score"] for row in rows]
            label = "Best fitness" if self.algorithm == "evolution" else "Greedy evaluation score (every 10 episodes)"
        elif metric == "progress":
            values = [row["progress"] * 100 for row in rows]
            label = "Track progress %"
        elif metric == "completion":
            values = [row["completion"] * 100 for row in rows]
            label = "Lap completion %"
        elif metric == "crashes":
            values = [row["crashes"] * 100 for row in rows]
            label = "Crash rate %"
        elif metric == "lap":
            values = [row["best_lap"] / 60 if row["best_lap"] else None for row in rows]
            label = "Best lap seconds"
        else:
            values = [row["reward"] or 0 for row in rows]
            label = "Episode reward"
        return rows, values, label

    def _draw_chart(self):
        box(self.screen, pygame.Rect(RX, 370, RW, 240))
        text(self.screen, self.bold, "TRAINING HISTORY", TEXT, RX + 18, 384)
        metrics = [("score", "Fitness" if self.algorithm == "evolution" else "Greedy score"),
                   ("progress", "Track progress"), ("completion", "Lap completion"),
                   ("crashes", "Crash rate"), ("lap", "Best lap time")]
        if self.algorithm == "dqn":
            metrics.append(("reward", "Training reward"))
        selected = self.chart_metric if self.chart_metric in dict(metrics) else "score"
        position = [key for key, _ in metrics].index(selected) + 1
        self.button("‹", (RX + 18, 415, 38, 36), lambda: self.cycle_metric(-1))
        pygame.draw.rect(self.screen, PANEL2, (RX + 63, 415, 209, 36), border_radius=7)
        text(self.screen, self.font, dict(metrics)[selected], TEXT, RX + 75, 422, 190)
        self.button("›", (RX + 279, 415, 38, 36), lambda: self.cycle_metric(1))
        text(self.screen, self.small, f"{position} / {len(metrics)} metrics", MUTED, RX + 329, 423)
        rows, values, label = self._metric_values()
        valid = [v for v in values if v is not None]
        latest = f"{valid[-1]:.1f}" if valid else "—"
        text(self.screen, self.small, f"{label}   ·   latest {latest}", MUTED, RX + 18, 455, 505)
        plot = pygame.Rect(RX + 49, 480, RW - 67, 102)
        pygame.draw.rect(self.screen, (15, 25, 36), plot)
        for fraction in (0, 0.5, 1):
            y = plot.bottom - fraction * plot.height
            pygame.draw.line(self.screen, (45, 60, 75), (plot.left, y), (plot.right, y))
        if not valid:
            prompt = "A finished greedy evaluation will appear here." if self.algorithm == "dqn" and selected == "score" else "Finish a run to start this graph."
            text(self.screen, self.small, prompt, MUTED, plot.x + 74, plot.y + 40)
            return
        low, high = min(valid), max(valid)
        if low == high:
            low -= 1
            high += 1
        padding = (high - low) * 0.12
        low -= padding
        high += padding
        text(self.screen, self.tiny, f"{high:.1f}", MUTED, RX + 4, plot.y - 4)
        text(self.screen, self.tiny, f"{low:.1f}", MUTED, RX + 4, plot.bottom - 13)
        def coords(sequence):
            return [(plot.left + i * plot.width / max(1, len(values) - 1),
                     plot.bottom - (value - low) / (high - low) * plot.height)
                    for i, value in enumerate(sequence) if value is not None]
        points = coords(values)
        if len(points) > 1:
            pygame.draw.lines(self.screen, BLUE, False, points, 2)
        if len(values) >= 3 and all(v is not None for v in values):
            smoothed = [float(np.mean(values[max(0, i - 9):i + 1])) for i in range(len(values))]
            pygame.draw.lines(self.screen, GREEN, False, coords(smoothed), 2)
        if self.algorithm == "dqn" and selected == "score" and len(values) > 1:
            best_so_far = list(np.maximum.accumulate(values))
            pygame.draw.lines(self.screen, ORANGE, False, coords(best_so_far), 2)
            legend = "Each eval     10-eval mean     Best so far"
        else:
            legend = "Each run     Trailing 10-run mean"
        pygame.draw.line(self.screen, BLUE, (RX + 18, 595), (RX + 34, 595), 3)
        text(self.screen, self.tiny, legend.split("     ")[0], MUTED, RX + 39, 587)
        pygame.draw.line(self.screen, GREEN, (RX + 122, 595), (RX + 138, 595), 3)
        text(self.screen, self.tiny, legend.split("     ")[1], MUTED, RX + 143, 587)
        if self.algorithm == "dqn" and selected == "score":
            pygame.draw.line(self.screen, ORANGE, (RX + 278, 595), (RX + 294, 595), 3)
            text(self.screen, self.tiny, "Best so far", MUTED, RX + 299, 587)
        unit = "evals" if self.algorithm == "dqn" and selected == "score" else "runs"
        text(self.screen, self.tiny, f"{len(rows)} {unit}", MUTED, RX + 451, 587)

    def _draw_settings(self):
        box(self.screen, pygame.Rect(RX, 625, RW, 260))
        text(self.screen, self.bold, "RUN STATUS", TEXT, RX + 18, 637)
        self.button("Sensors  R", (RX + 402, 632, 118, 26), self.toggle_sensor_panel,
                    active=self.sensor_panel, disabled=self.view not in ("train", "compare"))
        trainer = self.trainer
        if self.algorithm == "evolution":
            run_text = (f"Generation {trainer.generation}  ·  {trainer.active_count}/{len(trainer.population)} driving"
                        f"  ·  E{trainer.index + 1} selected")
            extra = (f"Best fitness {trainer.best_score:.1f}" if math.isfinite(trainer.best_score)
                     else "Loaded AI; untested here" if trainer.best_network else "No finished car yet")
        else:
            run_text = f"Episode {trainer.episode}  ·  ε {trainer.epsilon:.2f}  ·  {len(trainer.memory)} memories"
            extra = (f"Best greedy {trainer.best_score:.1f}" if math.isfinite(trainer.best_score)
                     else "Loaded AI; untested here" if trainer.best_network
                     else "Greedy eval every 10 runs")
        text(self.screen, self.small, run_text, GREEN, RX + 18, 662, 328)
        text(self.screen, self.small, extra, TEXT, RX + 352, 662, 169)
        car = trainer.car if self.view in ("train", "compare") else self.opponent if self.view == "race" else None
        if car:
            progress = min(100, car.max_progress / self.track.total_length * 100)
            status = f"Lap {car.laps}  ·  next gate {car.next_gate + 1}/{len(self.track.checkpoints)}"
        elif self.view == "replay" and self.replay:
            progress = self.replay_index / max(1, len(self.replay["frames"]) - 1) * 100
            status = f"Replay frame {self.replay_index + 1}/{len(self.replay['frames'])}"
        else:
            progress = 0
            status = "Shared track, sensors, physics, and scoring"
        text(self.screen, self.small, status, MUTED, RX + 18, 685, 382)
        text(self.screen, self.mono, f"{progress:.0f}%", GREEN, RX + 471, 683)
        pygame.draw.rect(self.screen, (35, 53, 66), (RX + 18, 708, RW - 36, 5), border_radius=3)
        if progress:
            pygame.draw.rect(self.screen, GREEN, (RX + 18, 708, round((RW - 36) * progress / 100), 5), border_radius=3)
        text(self.screen, self.tiny, "TRAINING SETTINGS", BLUE, RX + 18, 719)
        text(self.screen, self.tiny, f"Sensors: {self.sensors.label()}", MUTED, RX + 141, 719, 380)
        rows = [("Seed", "seed", "Same random start for a repeatable run"),
                ("Max steps", "max_steps", "Time limit per car or episode")]
        if self.algorithm == "evolution":
            rows += [("Population", "population", "Drivers tested each generation"),
                     ("Mutation", "mutation", "Size of random weight changes")]
        else:
            rows += [("Learning rate", "learning_rate", "Size of each network update"),
                     ("Explore decay", "epsilon_decay", "How quickly random actions fade"),
                     ("Batch size", "batch_size", "Past steps used for each update")]
        for i, (label, attr, hint) in enumerate(rows):
            y = 740 + i * 28
            value = getattr(self.settings, attr)
            shown = f"{value:.4f}" if isinstance(value, float) else str(value)
            text(self.screen, self.small, label, TEXT, RX + 18, y + 3)
            text(self.screen, self.tiny, hint, MUTED, RX + 141, y + 5, 199)
            self.button("−", (RX + 352, y, 32, 28),
                        lambda attr=attr: self.change_setting(attr, -1),
                        disabled=self.view not in ("train", "compare"))
            pygame.draw.rect(self.screen, PANEL2, (RX + 391, y, 89, 28), border_radius=6)
            rendered = self.mono.render(shown, True, GREEN)
            self.screen.blit(rendered, rendered.get_rect(center=(RX + 435, y + 14)))
            self.button("+", (RX + 488, y, 32, 28),
                        lambda attr=attr: self.change_setting(attr, 1),
                        disabled=self.view not in ("train", "compare"))
        if self.algorithm == "evolution":
            text(self.screen, self.tiny, "Changing a setting restarts both learners.", MUTED, RX + 18, 858)

    def _draw_bottom(self):
        box(self.screen, pygame.Rect(20, 739, 900, 146))
        if self.view in ("train", "compare"):
            self.button("Pause" if not self.paused else "Resume", (36, 755, 90, 38),
                        lambda: setattr(self, "paused", not self.paused), accent=True)
            self.button(f"Speed ×{self.current_speed()}", (133, 755, 100, 38),
                        lambda: setattr(self, "speed_index", (self.speed_index + 1) % len(self.speeds)))
            self.button("Reset training", (240, 755, 115, 38), self.reset_trainers)
            self.button("Save model", (362, 755, 105, 38), self.save_model)
            self.button("Load model", (474, 755, 105, 38), self.load_model)
            self.button("Best replay", (586, 755, 100, 38), self.replay_best)
            self.button("Last replay", (693, 755, 100, 38), self.replay_last)
            self.button("Save replay", (800, 755, 104, 38), self.save_replay)
            if self.view == "compare" or self.algorithm == "evolution":
                evolution = self.trainers["evolution"]
                self.button("← Watch car", (36, 802, 118, 36),
                            lambda: self.focus_evolution(-1))
                self.button("Watch car →", (163, 802, 118, 36),
                            lambda: self.focus_evolution(1))
                self.button("Pit wall  F", (290, 802, 116, 36), self.toggle_pit_board,
                            active=self.pit_board)
                self.button("What if?  I", (415, 802, 120, 36), self.open_sandbox)
                self.button("Wiring  N", (544, 802, 106, 36), self.toggle_wiring,
                            active=self.wiring)
                text(self.screen, self.small,
                     f"E{evolution.index + 1} selected · {evolution.active_count} active",
                     MUTED, 660, 810, 242)
                if self.view == "compare":
                    text(self.screen, self.small,
                         "Same track; each evolution tick advances every car, so sample budgets differ.",
                         MUTED, 38, 837)
                else:
                    text(self.screen, self.small,
                         "Every colored car is a separate neural network in the same generation.",
                         MUTED, 38, 837)
            else:
                self.button("What if?  I", (36, 802, 133, 36), self.open_sandbox)
                self.button("Wiring  N", (178, 802, 106, 36), self.toggle_wiring,
                            active=self.wiring)
                text(self.screen, self.small,
                     "Probe one frozen DQN decision, or open the wiring diagram.",
                     MUTED, 296, 810, 604)
                text(self.screen, self.small,
                     "Use Compare models to train DQN beside the evolution population. Space pauses.",
                     MUTED, 296, 832, 604)
        elif self.view == "race":
            self.button("Restart race", (36, 755, 130, 38), self.start_race, accent=True)
            self.button("Pause" if not self.paused else "Resume", (176, 755, 110, 38),
                        lambda: setattr(self, "paused", not self.paused))
            self.button("Back to training", (296, 755, 155, 38),
                        lambda: setattr(self, "view", "train"))
            control_name = self.race_controls[self.race_control_index][0]
            self.button(f"Steering: {control_name}", (461, 755, 157, 38),
                        self.cycle_race_controls)
            self.button("Car garage  C", (628, 755, 139, 38), self.toggle_car_panel)
            text(self.screen, self.small,
                 "Hold ↑/W for gas · tap or hold ←/A and →/D to steer · ↓/S brakes",
                 MUTED, 38, 808, 620)
            text(self.screen, self.mono,
                 f"V {self.human.speed:3.1f}   STEER {self.race_steer:+.2f}   GAS {self.race_throttle:.2f}",
                 BLUE, 655, 806, 245)
            if self.human and self.opponent:
                text(self.screen, self.small,
                     f"You: {self.human.max_progress / self.track.total_length * 100:.0f}% / {self.human.laps} laps"
                     f"    AI: {self.opponent.max_progress / self.track.total_length * 100:.0f}% / {self.opponent.laps} laps",
                     TEXT, 38, 832)
        elif self.view == "editor":
            buttons = [("Point tool", lambda: setattr(self, "editor_tool", "points")),
                       ("Gate tool", lambda: setattr(self, "editor_tool", "gates")),
                       ("Set start", self.set_start),
                       ("Clear", lambda: self._clear_editor()),
                       ("Apply track", self.apply_editor),
                       ("Save", self.save_track), ("Open", self.open_track)]
            x = 35
            for label, action in buttons:
                width = 116 if label == "Apply track" else 103
                self.button(label, (x, 754, width, 37), action,
                            accent=label == "Apply track",
                            active=(label == "Point tool" and self.editor_tool == "points") or
                                   (label == "Gate tool" and self.editor_tool == "gates"))
                x += width + 8
            self.button("Undo point", (35, 801, 118, 32),
                        lambda: self._undo_editor())
            self.button("Width −", (163, 801, 100, 32),
                        lambda: self._width_editor(-8))
            self.button("Width +", (273, 801, 100, 32),
                        lambda: self._width_editor(8))
            text(self.screen, self.small,
                 f"{len(self.editor_points)} points · width {self.editor_width:.0f} · "
                 f"{len(self.editor_gates) if self.editor_gates else 'auto'} gates",
                 TEXT, 390, 807)
            text(self.screen, self.small,
                 "Select a point, then Set start. The start gate is point 1; extra gates can be placed on the road.",
                 MUTED, 36, 844)
        elif self.view == "replay":
            self.button("Play" if self.paused else "Pause", (35, 754, 105, 38),
                        lambda: setattr(self, "paused", not self.paused), accent=True)
            self.button("−60", (150, 754, 78, 38), lambda: self._step_replay(-60))
            self.button("−1", (238, 754, 78, 38), lambda: self._step_replay(-1))
            self.button("+1", (326, 754, 78, 38), lambda: self._step_replay(1))
            self.button("+60", (414, 754, 78, 38), lambda: self._step_replay(60))
            self.button("Open replay", (502, 754, 126, 38), self.open_replay)
            self.button("Train", (638, 754, 100, 38), lambda: setattr(self, "view", "train"))
            self.button("Save replay", (748, 754, 130, 38), self.save_replay)
            if self.replay:
                text(self.screen, self.small, self.replay["label"], TEXT, 38, 807)
                text(self.screen, self.small,
                     f"Frame {self.replay_index + 1}/{len(self.replay['frames'])}  ·  "
                     "Sensors and network values are those recorded during the run.", MUTED, 38, 833)
        else:
            self.button("Back to training", (36, 755, 160, 38),
                        lambda: setattr(self, "view", "train"), accent=True)
            text(self.screen, self.small, "The README contains the full guided walkthrough and experiments.",
                 MUTED, 38, 811)
        message_color = RED if any(word in self.message.lower() for word in ("could not", "needs work", "unavailable")) else GREEN
        if pygame.time.get_ticks() - self.message_time < 13000:
            text(self.screen, self.small, self.message, message_color, 36, 862, 855)

    def _draw_guide(self):
        box(self.screen, pygame.Rect(70, 123, 800, 550), PANEL)
        text(self.screen, self.title, "HOW A DRIVER LEARNS", TEXT, 99, 150)
        lines = [
            ("1  SEE", f"{self.sensors.label()}. Press R to change what the car senses."),
            ("2  DECIDE", "The network transforms inputs through 16 hidden neurons into five action values."),
            ("3  ACT", "The car steers, accelerates, coasts, or brakes under shared physics."),
            ("4  FEEDBACK", "New forward progress and ordered gates earn credit; crashes and time cost points."),
            ("EVOLUTION", "A whole generation drives together. Select a car to inspect its own network."),
            ("DQN", "One network learns action values from past steps; exploration slowly decreases."),
            ("EXPLORE", "Compare trains both methods. F pit wall · I what-if · N wiring · R sensors."),
            ("EVALUATE", "Compare on a track the learner has not trained on. Training score alone can mislead."),
        ]
        for i, (heading, body) in enumerate(lines):
            y = 205 + i * 60
            text(self.screen, self.bold, heading, GREEN if i < 4 else BLUE, 100, y)
            text(self.screen, self.small, body, TEXT, 258, y + 1, 585)

    def _clear_editor(self):
        self.editor_points = []
        self.editor_gates = []
        self.editor_name = "Custom Loop"
        self.editor_selected = None
        self.status("Canvas cleared. Click at least five points to draw a closed centerline.")

    def _undo_editor(self):
        if self.editor_points:
            self.editor_points.pop()
            self.editor_gates = []
            self.editor_selected = None

    def _width_editor(self, delta):
        self.editor_width = max(55, min(160, self.editor_width + delta))

    def _step_replay(self, amount):
        if self.replay:
            self.replay_index = max(0, min(len(self.replay["frames"]) - 1,
                                           self.replay_index + amount))
            self.paused = True

    def _preview_track(self):
        if len(self.editor_points) < 5:
            return None
        key = (tuple(self.editor_points), self.editor_width, tuple(self.editor_gates))
        if key != self._editor_preview_key:
            self._editor_preview_key = key
            try:
                self._editor_preview = Track(self.editor_points, self.editor_width, "Preview",
                                             sorted(self.editor_gates) if self.editor_gates else None)
            except ValueError:
                self._editor_preview = None
        return self._editor_preview

    def _editor_click(self, position, mouse_button):
        x, y = position[0] - OX, position[1] - OY
        if not (0 <= x < WORLD_W and 0 <= y < WORLD_H):
            return
        if self.editor_tool == "points":
            nearest = min(range(len(self.editor_points)),
                          key=lambda i: math.dist((x, y), self.editor_points[i])) if self.editor_points else None
            near = nearest is not None and math.dist((x, y), self.editor_points[nearest]) < 17
            if mouse_button == 3 and near:
                self.editor_points.pop(nearest)
                self.editor_gates = []
                self.editor_selected = None
            elif mouse_button == 1 and near:
                self.editor_selected = nearest
                self.dragging = True
            elif mouse_button == 1:
                margin = self.editor_width / 2 + 12
                if margin <= x <= WORLD_W - margin and margin <= y <= WORLD_H - margin:
                    self.editor_points.append((x, y))
                    self.editor_selected = len(self.editor_points) - 1
                    self.editor_gates = []
                else:
                    self.status("Place points further from the canvas edge.")
        elif len(self.editor_points) >= 5:
            try:
                track = self._preview_track()
                if track is None:
                    self.status("Finish a valid loop before placing gates.")
                    return
                if mouse_button == 1:
                    fraction = track.nearest_progress(x, y) / track.total_length
                    if not track.road_at(x, y):
                        self.status("Click on the road to place a gate.")
                    else:
                        candidate = sorted(set(self.editor_gates or track.gate_fractions) | {fraction})
                        Track(self.editor_points, self.editor_width, "Preview", candidate)
                        self.editor_gates = candidate
                elif mouse_button == 3 and self.editor_gates:
                    candidates = [(i, math.dist((x, y), gate[:2]))
                                  for i, gate in enumerate(track.checkpoints) if i > 0]
                    index, distance = min(candidates, key=lambda row: row[1])
                    if distance < 22 and len(self.editor_gates) > 4:
                        self.editor_gates.pop(index)
            except ValueError as exc:
                self.status(str(exc))

    def _select_car_on_track(self, position):
        x, y = position[0] - OX, position[1] - OY
        if not (0 <= x < WORLD_W and 0 <= y < WORLD_H):
            return
        evolution = self.trainers["evolution"]
        candidates = sorted(((math.dist((x, y), (car.x, car.y)), i)
                             for i, car in enumerate(evolution.cars)))
        if self.view == "compare":
            learner = self.trainers["dqn"].car
            learner_distance = math.dist((x, y), (learner.x, learner.y))
            if learner_distance < 20 and learner_distance + 4 < candidates[0][0]:
                self.algorithm = "dqn"
                self.status("Inspecting the reinforcement learning driver.")
                return
        nearby = [i for distance, i in candidates if distance < 20]
        if nearby:
            selected = (nearby[(nearby.index(evolution.index) + 1) % len(nearby)]
                        if evolution.index in nearby and len(nearby) > 1 else nearby[0])
            evolution.focus(selected)
            self.algorithm = "evolution"
            self.status(f"Inspecting evolution model E{selected + 1}.")

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.sandbox is not None and pygame.Rect(130, 454, 650, 32).collidepoint(event.pos):
                    self.sandbox_dragging = True
                    self._set_sandbox_slider(event.pos[0])
                    return True
                if self.sandbox is not None and not pygame.Rect(100, 151, 740, 526).collidepoint(event.pos):
                    return True
                for rect, action in reversed(self.buttons):
                    if rect.collidepoint(event.pos):
                        action()
                        return True
                if self.sandbox is not None or (self.pit_board and pygame.Rect(90, 143, 760, 523).collidepoint(event.pos)):
                    return True
                if self.car_panel:
                    return True
                # Clicks inside the wiring or sensor overlays must not select a car underneath.
                if ((self.wiring or self.sensor_panel and self.view in ("train", "compare"))
                        and pygame.Rect(100, 151, 740, 526).collidepoint(event.pos)):
                    return True
            if self.view == "editor":
                self._editor_click(event.pos, event.button)
            elif event.button == 1 and (self.view == "compare" or
                                       self.view == "train" and self.algorithm == "evolution"):
                self._select_car_on_track(event.pos)
        if event.type == pygame.MOUSEMOTION and self.sandbox_dragging:
            self._set_sandbox_slider(event.pos[0])
        if event.type == pygame.MOUSEMOTION and self.view == "editor" and self.dragging:
            x, y = event.pos[0] - OX, event.pos[1] - OY
            margin = self.editor_width / 2 + 12
            if self.editor_selected is not None:
                self.editor_points[self.editor_selected] = (max(margin, min(WORLD_W - margin, x)),
                                                           max(margin, min(WORLD_H - margin, y)))
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            self.sandbox_dragging = False
        if event.type == pygame.KEYDOWN:
            if self.car_panel:
                if event.key in (pygame.K_ESCAPE, pygame.K_c):
                    self.toggle_car_panel()
                return True
            if self.sandbox is not None:
                if event.key in (pygame.K_ESCAPE, pygame.K_i):
                    self._close_sandbox()
                return True
            if self.wiring and event.key in (pygame.K_ESCAPE, pygame.K_n):
                self.toggle_wiring()
                return True
            if self.sensor_panel and event.key in (pygame.K_ESCAPE, pygame.K_r):
                self.toggle_sensor_panel()
                return True
            if event.key == pygame.K_r and self.view in ("train", "compare"):
                self.toggle_sensor_panel()
                return True
            if event.key == pygame.K_c:
                self.toggle_car_panel()
                return True
            if event.key == pygame.K_n and self.view in ("train", "compare", "race", "replay"):
                self.toggle_wiring()
            elif event.key == pygame.K_f and (self.view == "compare" or
                                             self.view == "train" and self.algorithm == "evolution"):
                self.toggle_pit_board()
            elif event.key == pygame.K_i and self.view in ("train", "compare"):
                self.open_sandbox()
            elif event.key == pygame.K_SPACE and self.view in ("train", "compare", "race", "replay"):
                self.paused = not self.paused
            elif event.key == pygame.K_ESCAPE and self.view != "train":
                self.view = "train"
            elif event.key == pygame.K_s and self.view == "editor":
                self.set_start()
        return True

    def draw(self):
        self.buttons = []
        self.screen.fill(BG)
        self._draw_header()
        self._draw_world()
        if self.view == "guide":
            self._draw_guide()
        self._draw_inspector()
        self._draw_chart()
        self._draw_settings()
        self._draw_bottom()
        self._draw_pit_board()
        self._draw_wiring()
        self._draw_sensor_panel()
        self._draw_car_panel()
        self._draw_sandbox()
        pygame.display.flip()

    def run(self, smoke=False, screenshot=None):
        alive, frames = True, 0
        while alive:
            for event in pygame.event.get():
                alive = self.handle_event(event)
                if not alive:
                    break
            if not alive:
                break
            self.tick()
            self.draw()
            frames += 1
            if smoke and frames >= 3:
                if screenshot:
                    pygame.image.save(self.screen, screenshot)
                break
            self.clock.tick(60)
        self._save_preferences()
        pygame.quit()


def main():
    smoke = "--smoke" in sys.argv
    screenshot = None
    if "--screenshot" in sys.argv:
        index = sys.argv.index("--screenshot")
        screenshot = sys.argv[index + 1]
    App().run(smoke, screenshot)


if __name__ == "__main__":
    main()
