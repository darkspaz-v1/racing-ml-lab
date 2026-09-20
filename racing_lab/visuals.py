"""Small top-down car artwork drawn in Pygame, aligned with the physics body."""

from __future__ import annotations

import math

import pygame


class CarPainter:
    """Cache rotated car sprites so a whole evolution population is inexpensive."""

    def __init__(self):
        self.bases: dict[tuple, pygame.Surface] = {}
        self.rotated: dict[tuple, pygame.Surface] = {}

    def _base(self, color: tuple[int, int, int], crashed: bool) -> pygame.Surface:
        key = (color, crashed)
        if key in self.bases:
            return self.bases[key]
        car = pygame.Surface((42, 30), pygame.SRCALPHA)
        body = (197, 74, 80) if crashed else color
        # The visible body is about 26 × 16 pixels, matching the collision box.
        pygame.draw.ellipse(car, (0, 0, 0, 75), (4, 7, 34, 18))
        for rect in ((9, 6, 8, 5), (26, 6, 8, 5), (9, 19, 8, 5), (26, 19, 8, 5)):
            pygame.draw.rect(car, (9, 16, 24), rect, border_radius=2)
            pygame.draw.line(car, (67, 77, 84), rect[:2], (rect[0] + rect[2], rect[1]), 1)
        pygame.draw.rect(car, (8, 18, 26), (7, 7, 28, 16), border_radius=5)
        pygame.draw.rect(car, body, (8, 8, 26, 14), border_radius=4)
        # Roof and glass make the front (right) end immediately recognizable.
        roof = (max(0, body[0] - 32), max(0, body[1] - 32), max(0, body[2] - 32))
        pygame.draw.polygon(car, roof, ((16, 10), (25, 10), (28, 15), (25, 20), (16, 20), (13, 15)))
        pygame.draw.polygon(car, (48, 85, 102), ((23, 11), (26, 12), (28, 15), (26, 18), (23, 19)))
        pygame.draw.line(car, (111, 165, 179), (25, 11), (27, 14), 1)
        pygame.draw.polygon(car, (30, 58, 72), ((15, 11), (18, 11), (18, 19), (15, 19), (13, 15)))
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
             color: tuple[int, int, int], alpha: int = 255, crashed: bool = False) -> None:
        degrees = round(math.degrees(angle) / 5) * 5 % 360
        key = (color, crashed, degrees, alpha)
        if key not in self.rotated:
            sprite = pygame.transform.rotate(self._base(color, crashed), -degrees)
            if alpha != 255:
                sprite.set_alpha(alpha)
            self.rotated[key] = sprite
        sprite = self.rotated[key]
        surface.blit(sprite, sprite.get_rect(center=(round(x), round(y))))

    def draw_preview(self, surface: pygame.Surface, center: tuple[int, int],
                     color: tuple[int, int, int], crashed: bool = False) -> None:
        """Magnify the same sprite for the model detail panel."""
        sprite = pygame.transform.scale(self._base(color, crashed), (126, 90))
        surface.blit(sprite, sprite.get_rect(center=center))
