"""Small auditable numerical primitives used by operator certification."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence


Vector = Sequence[float]


def dot(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vector lengths differ")
    return math.fsum(float(a) * float(b) for a, b in zip(left, right))


def central_directional_derivative(
    fn: Callable[[list[float]], float], point: Vector, direction: Vector, epsilon: float
) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if len(point) != len(direction):
        raise ValueError("point and direction lengths differ")
    plus = [float(x) + epsilon * float(v) for x, v in zip(point, direction)]
    minus = [float(x) - epsilon * float(v) for x, v in zip(point, direction)]
    return (float(fn(plus)) - float(fn(minus))) / (2.0 * epsilon)


def relative_error(observed: float, expected: float, *, floor: float = 1e-12) -> float:
    return abs(observed - expected) / max(abs(expected), floor)


def bilinear_symmetry_error(
    apply: Callable[[Vector], Vector], left: Vector, right: Vector
) -> float:
    xy = dot(left, apply(right))
    yx = dot(right, apply(left))
    return abs(xy - yx) / max(abs(xy), abs(yx), 1e-12)


def linearity_error(
    apply: Callable[[Vector], Vector], left: Vector, right: Vector, alpha: float
) -> float:
    combined = [float(x) + alpha * float(y) for x, y in zip(left, right)]
    actual = list(apply(combined))
    expected = [float(a) + alpha * float(b) for a, b in zip(apply(left), apply(right))]
    numerator = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(actual, expected)))
    denominator = max(math.sqrt(math.fsum(b * b for b in expected)), 1e-12)
    return numerator / denominator
