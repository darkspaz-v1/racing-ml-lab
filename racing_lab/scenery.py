"""Paint a track as a motor-racing circuit: asphalt, kerbs, gravel, grass, and barriers.

Everything is drawn from the same centreline distance the physics uses, so the
painted road edge is exactly where a car leaves the road and crashes. Kerbs and
white lines sit inside that edge; gravel and grass are outside it.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from .track import WORLD_H, WORLD_W, Track

GRASS = ((39, 82, 50), (44, 91, 56))
ASPHALT = (57, 61, 67)
LINE = (226, 229, 225)
KERB_RED = (196, 56, 54)
KERB_WHITE = (231, 231, 227)
GRAVEL = (150, 131, 97)
BARRIER = (168, 175, 181)
TYRE = ((34, 35, 38), (206, 64, 60))
CHECKER = ((20, 22, 25), (238, 238, 234))

CORNER_TURN = math.radians(15)   # weighted heading change that counts as a corner
CORNER_WINDOW = 60               # px either side of a point along the centreline
GRAVEL_DEPTH = 30
BARRIER_GAP = 40


def _distance_field(track: Track, reach: float):
    """Per pixel: distance to the centreline, arc position, and signed side."""
    distance = np.full((WORLD_H, WORLD_W), np.inf)
    arc = np.zeros((WORLD_H, WORLD_W))
    side = np.zeros((WORLD_H, WORLD_W))
    cumulative = 0.0
    for ((ax, ay), (bx, by)), length in zip(track.segments, track.lengths):
        x0, x1 = max(0, int(min(ax, bx) - reach)), min(WORLD_W, int(max(ax, bx) + reach + 1))
        y0, y1 = max(0, int(min(ay, by) - reach)), min(WORLD_H, int(max(ay, by) + reach + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = bx - ax, by - ay
        t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / (length * length), 0, 1)
        ox, oy = xx - (ax + t * dx), yy - (ay + t * dy)
        here = np.sqrt(ox * ox + oy * oy)
        closer = here < distance[y0:y1, x0:x1]
        distance[y0:y1, x0:x1][closer] = here[closer]
        arc[y0:y1, x0:x1][closer] = (cumulative + t * length)[closer]
        side[y0:y1, x0:x1][closer] = ((ox * -dy + oy * dx) / length)[closer]
        cumulative += length
    return distance, arc, side


def _turn_profile(track: Track, step: float = 2.0) -> np.ndarray:
    """Signed heading change around each arc sample, weighted toward the sample.

    For a polyline the heading only changes at vertices, so this is a sum of
    vertex turn angles, each weighted by a triangle that falls to zero at the
    window's edge. The triangle makes corner zones (and so gravel traps) taper
    instead of ending in a hard cut. Positive is a right-hand turn on screen.
    """
    headings = [math.atan2(by - ay, bx - ax) for (ax, ay), (bx, by) in track.segments]
    turns = np.array([(headings[i] - headings[i - 1] + math.pi) % (2 * math.pi) - math.pi
                      for i in range(len(headings))])
    vertex_arc = np.concatenate([[0.0], np.cumsum(track.lengths)[:-1]])
    samples = np.arange(0, track.total_length, step)
    total = track.total_length
    offset = (vertex_arc[None, :] - samples[:, None] + total / 2) % total - total / 2
    return np.clip(1 - np.abs(offset) / CORNER_WINDOW, 0, 1) @ turns


def render_track(track: Track) -> pygame.Surface:
    half = track.width / 2
    reach = half + BARRIER_GAP + 8
    distance, arc, side = _distance_field(track, reach)
    turn = _turn_profile(track)
    step = track.total_length / len(turn)
    corner_turn = turn[np.clip((arc / step).astype(int), 0, len(turn) - 1)]
    corner = np.abs(corner_turn) > CORNER_TURN
    outside = side * np.sign(corner_turn) < 0

    yy, xx = np.mgrid[0:WORLD_H, 0:WORLD_W]
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, (WORLD_H, WORLD_W))

    stripes = ((xx * 0.8 + yy * 0.6) // 34).astype(int) % 2
    image = np.where(stripes[..., None] == 0, np.array(GRASS[0]), np.array(GRASS[1])).astype(float)
    image += noise[..., None] * 2.2

    # Deeper gravel where the corner is sharper, tapering to nothing at its ends.
    depth = GRAVEL_DEPTH * np.clip((np.abs(corner_turn) - CORNER_TURN) / CORNER_TURN, 0, 1)
    gravel = corner & outside & (distance > half) & (distance <= half + depth)
    speckle = rng.random((WORLD_H, WORLD_W))
    image[gravel] = np.array(GRAVEL) + (speckle[gravel, None] - 0.5) * 34

    barrier = (distance > half + BARRIER_GAP - 2) & (distance <= half + BARRIER_GAP + 2)
    tyres = barrier & corner
    image[barrier & ~corner] = BARRIER
    tyre_colour = ((arc // 9).astype(int) % 2)[tyres]
    image[tyres] = np.where(tyre_colour[:, None] == 0, np.array(TYRE[0]), np.array(TYRE[1]))

    # The road itself. Every drivable pixel is fully asphalt; the one-pixel soft
    # edge lies just outside the physics boundary, never on the road.
    road = np.array(ASPHALT) + noise[..., None] * 3.0
    cover = np.clip(half + 1 - distance, 0, 1)[..., None]
    image = image * (1 - cover) + road * cover

    on_road = distance <= half
    kerb = on_road & corner & (distance > half - 11)
    kerb_colour = ((arc // 13).astype(int) % 2)[kerb]
    image[kerb] = np.where(kerb_colour[:, None] == 0, np.array(KERB_RED), np.array(KERB_WHITE))
    line = on_road & ~kerb & (distance > half - 5.5) & (distance <= half - 2.5)
    image[line] = LINE

    # Chequered start/finish line across gate 0.
    start = track.checkpoints[0][4]
    along = (arc - start + track.total_length / 2) % track.total_length - track.total_length / 2
    finish = on_road & (np.abs(along) <= 7)
    squares = ((np.floor((along + 7) / 7) + np.floor((side + half) / 7)).astype(int) % 2)[finish]
    image[finish] = np.where(squares[:, None] == 0, np.array(CHECKER[0]), np.array(CHECKER[1]))

    pixels = np.clip(image, 0, 255).astype(np.uint8)
    return pygame.surfarray.make_surface(pixels.swapaxes(0, 1))
