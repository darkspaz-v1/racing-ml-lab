"""Small top-down car artwork drawn in Pygame, aligned with the physics body."""

from __future__ import annotations

import math

import pygame

CAR_STYLES = {
    "touring": ("Touring", "Balanced road-racing body"),
    "formula": ("Formula", "Open wheels and slim cockpit"),
    "rally": ("Rally", "Boxy body with a racing stripe"),
    "prototype": ("Prototype", "Low, wide endurance shape"),
}

CAR_COLORS = (
    (255, 181, 89), (81, 222, 152), (100, 181, 247), (177, 139, 237),
    (255, 106, 110), (245, 226, 98), (237, 245, 249), (66, 214, 220),
)


class CarPainter:
    """Cache rotated car sprites so a whole evolution population is inexpensive."""

    def __init__(self):
        self.bases: dict[tuple, pygame.Surface] = {}
        self.rotated: dict[tuple, pygame.Surface] = {}

    def _base(self, color: tuple[int, int, int], crashed: bool,
              style: str = "touring") -> pygame.Surface:
        if style not in CAR_STYLES:
            style = "touring"
        key = (color, crashed, style)
        if key in self.bases:
            return self.bases[key]
        car = pygame.Surface((42, 30), pygame.SRCALPHA)
        body = (197, 74, 80) if crashed else color
        pygame.draw.ellipse(car, (0, 0, 0, 75), (4, 7, 34, 18))
        wheel_rects = ((9, 6, 8, 5), (26, 6, 8, 5), (9, 19, 8, 5), (26, 19, 8, 5))
        if style == "formula":
            wheel_rects = ((8, 4, 8, 6), (27, 4, 8, 6), (8, 20, 8, 6), (27, 20, 8, 6))
        for rect in wheel_rects:
            pygame.draw.rect(car, (9, 16, 24), rect, border_radius=2)
            pygame.draw.line(car, (67, 77, 84), rect[:2], (rect[0] + rect[2], rect[1]), 1)
        roof = (max(0, body[0] - 32), max(0, body[1] - 32), max(0, body[2] - 32))
        if style == "formula":
            pygame.draw.rect(car, body, (10, 12, 25, 6), border_radius=3)
            pygame.draw.polygon(car, body, ((16, 9), (29, 12), (35, 15), (29, 18), (16, 21)))
            pygame.draw.ellipse(car, (35, 65, 82), (19, 10, 10, 10))
            pygame.draw.rect(car, roof, (6, 8, 3, 14))
            pygame.draw.rect(car, roof, (34, 7, 3, 16))
        elif style == "rally":
            pygame.draw.rect(car, (8, 18, 26), (6, 7, 30, 16), border_radius=3)
            pygame.draw.rect(car, body, (7, 8, 28, 14), border_radius=3)
            pygame.draw.rect(car, roof, (14, 9, 14, 12), border_radius=2)
            pygame.draw.rect(car, (230, 235, 229), (18, 8, 4, 14))
            pygame.draw.polygon(car, (48, 85, 102), ((25, 10), (29, 11), (31, 15), (29, 19), (25, 20)))
        elif style == "prototype":
            pygame.draw.polygon(car, (8, 18, 26), ((6, 10), (13, 6), (33, 8), (37, 15),
                                                   (33, 22), (13, 24), (6, 20)))
            pygame.draw.polygon(car, body, ((7, 11), (15, 7), (32, 9), (36, 15),
                                            (32, 21), (15, 23), (7, 19)))
            pygame.draw.ellipse(car, roof, (15, 9, 16, 12))
            pygame.draw.polygon(car, (48, 85, 102), ((23, 10), (29, 12), (31, 15),
                                                     (29, 18), (23, 20)))
        else:
            # The visible body is about 26 × 16 pixels, matching the collision box.
            pygame.draw.rect(car, (8, 18, 26), (7, 7, 28, 16), border_radius=5)
            pygame.draw.rect(car, body, (8, 8, 26, 14), border_radius=4)
            pygame.draw.polygon(car, roof, ((16, 10), (25, 10), (28, 15),
                                             (25, 20), (16, 20), (13, 15)))
            pygame.draw.polygon(car, (48, 85, 102), ((23, 11), (26, 12), (28, 15),
                                                     (26, 18), (23, 19)))
            pygame.draw.line(car, (111, 165, 179), (25, 11), (27, 14), 1)
            pygame.draw.polygon(car, (30, 58, 72), ((15, 11), (18, 11), (18, 19),
                                                    (15, 19), (13, 15)))
        pygame.draw.line(car, (227, 235, 226), (30, 10), (32, 10), 2)
        pygame.draw.line(car, (227, 235, 226), (30, 20), (32, 20), 2)
        pygame.draw.line(car, (255, 93, 88), (9, 10), (10, 10), 2)
        pygame.draw.line(car, (255, 93, 88), (9, 20), (10, 20), 2)
        pygame.draw.line(car, (191, 211, 216), (33, 13), (33, 17), 1)
        if crashed:
            pygame.draw.line(car, (255, 225, 120), (13, 8), (29, 22), 2)
            pygame.draw.line(car, (255, 225, 120), (29, 8), (13, 22), 2)
        self.bases[key] = car
        return car

    def draw(self, surface: pygame.Surface, x: float, y: float, angle: float,
             color: tuple[int, int, int], alpha: int = 255, crashed: bool = False,
             style: str = "touring") -> None:
        degrees = round(math.degrees(angle) / 5) * 5 % 360
        key = (color, crashed, style, degrees, alpha)
        if key not in self.rotated:
            sprite = pygame.transform.rotate(self._base(color, crashed, style), -degrees)
            if alpha != 255:
                sprite.set_alpha(alpha)
            self.rotated[key] = sprite
        sprite = self.rotated[key]
        surface.blit(sprite, sprite.get_rect(center=(round(x), round(y))))

    def draw_preview(self, surface: pygame.Surface, center: tuple[int, int],
                     color: tuple[int, int, int], crashed: bool = False,
                     style: str = "touring") -> None:
        """Magnify the same sprite for the model detail panel."""
        sprite = pygame.transform.scale(self._base(color, crashed, style), (126, 90))
        surface.blit(sprite, sprite.get_rect(center=center))
