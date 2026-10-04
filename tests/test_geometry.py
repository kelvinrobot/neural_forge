"""Tests for geometry helpers."""

import math

import numpy as np

from neural_forge.services import geometry as g


def test_joint_angle_right_angle() -> None:
    assert abs(g.joint_angle_degrees(np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0])) - 90.0) < 1e-9


def test_line_tilt_is_normalised() -> None:
    assert abs(g.line_tilt_degrees(np.array([0.0, 0.0]), np.array([10.0, 10.0])) - 45.0) < 1e-9
    assert abs(g.line_tilt_degrees(np.array([10.0, 10.0]), np.array([0.0, 0.0])) - 45.0) < 1e-9


def test_angle_from_vertical_sign() -> None:
    assert g.angle_from_vertical_degrees(np.array([1.0, 0.0]), np.array([0.0, 10.0])) > 0


def test_eye_aspect_ratio_of_open_eye() -> None:
    points = np.array([[0, 0], [3, -2], [7, -2], [10, 0], [7, 2], [3, 2]], dtype=float)
    assert abs(g.eye_aspect_ratio(points) - 0.4) < 1e-9


def test_polygon_area_of_square() -> None:
    assert g.polygon_area(np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)) == 4.0


def test_signed_distance_to_line() -> None:
    distance = g.signed_distance_to_line(np.array([0.0, 5.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert distance == 5.0


def test_projection_parameter() -> None:
    assert abs(g.projection_parameter(np.array([5.0, 3.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0])) - 0.5) < 1e-9


def test_confidence_ellipse_area_scales_with_spread() -> None:
    generator = np.random.default_rng(0)
    small = generator.normal(scale=1.0, size=(500, 2))
    large = generator.normal(scale=2.0, size=(500, 2))
    assert g.confidence_ellipse_area(large) > 3.0 * g.confidence_ellipse_area(small)


def test_asymmetry_ratio_bounds() -> None:
    assert g.asymmetry_ratio(1.0, 1.0) == 0.0
    assert g.asymmetry_ratio(0.0, 2.0) == 1.0
    assert g.asymmetry_ratio(0.0, 0.0) == 0.0


def test_fit_line_direction_vertical() -> None:
    points = np.array([[5.0, float(y)] for y in range(10)])
    _, direction = g.fit_line_direction(points)
    assert abs(abs(direction[1]) - 1.0) < 1e-9


def test_rotate_points() -> None:
    rotated = g.rotate_points(np.array([[1.0, 0.0]]), 90.0, np.array([0.0, 0.0]))
    assert np.allclose(rotated, [[0.0, 1.0]], atol=1e-9)
    assert math.isnan(g.safe_ratio(1.0, 0.0))
