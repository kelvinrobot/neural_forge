"""2D geometry helpers for landmark measurements."""

import math

import numpy as np

EPSILON = 1e-9


def as_vector(value: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert a point-like value to a float64 numpy array of its first two coordinates."""
    return np.asarray(value, dtype=np.float64)[:2]


def euclidean_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Distance between two points."""
    return float(np.linalg.norm(as_vector(first) - as_vector(second)))


def midpoint(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Midpoint of two points."""
    return (as_vector(first) + as_vector(second)) / 2.0


def vector_angle_degrees(first_vector: np.ndarray, second_vector: np.ndarray) -> float:
    """Unsigned angle in degrees between two vectors."""
    first = as_vector(first_vector)
    second = as_vector(second_vector)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < EPSILON:
        return float("nan")
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def signed_angle_degrees(from_vector: np.ndarray, to_vector: np.ndarray) -> float:
    """Signed angle in degrees rotating from one vector to another (image coordinates)."""
    first = as_vector(from_vector)
    second = as_vector(to_vector)
    cross = first[0] * second[1] - first[1] * second[0]
    dot = float(np.dot(first, second))
    return math.degrees(math.atan2(cross, dot))


def joint_angle_degrees(first: np.ndarray, vertex: np.ndarray, third: np.ndarray) -> float:
    """Angle at the vertex formed by the segments vertex-first and vertex-third."""
    return vector_angle_degrees(as_vector(first) - as_vector(vertex), as_vector(third) - as_vector(vertex))


def line_tilt_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Tilt of the line first->second relative to the image horizontal, in (-90, 90]."""
    delta = as_vector(second) - as_vector(first)
    angle = math.degrees(math.atan2(delta[1], delta[0]))
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


def angle_from_vertical_degrees(top: np.ndarray, bottom: np.ndarray) -> float:
    """Signed angle of the segment bottom->top from the image vertical; positive leans to image right."""
    delta = as_vector(top) - as_vector(bottom)
    return math.degrees(math.atan2(delta[0], -delta[1]))


def signed_distance_to_line(point: np.ndarray, line_point: np.ndarray, line_direction: np.ndarray) -> float:
    """Signed perpendicular distance from a point to a line given by a point and a direction."""
    direction = as_vector(line_direction)
    norm = float(np.linalg.norm(direction))
    if norm < EPSILON:
        return float("nan")
    unit = direction / norm
    offset = as_vector(point) - as_vector(line_point)
    return float(unit[0] * offset[1] - unit[1] * offset[0])


def projection_parameter(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Position of the point's projection along start->end, where 0 is start and 1 is end."""
    segment = as_vector(end) - as_vector(start)
    length_squared = float(np.dot(segment, segment))
    if length_squared < EPSILON:
        return float("nan")
    return float(np.dot(as_vector(point) - as_vector(start), segment) / length_squared)


def perpendicular_offset(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Signed perpendicular offset of a point from the line start->end, normalised by segment length."""
    segment = as_vector(end) - as_vector(start)
    length = float(np.linalg.norm(segment))
    if length < EPSILON:
        return float("nan")
    return signed_distance_to_line(point, start, segment) / length


def polygon_area(points: np.ndarray) -> float:
    """Area of a simple polygon using the shoelace formula."""
    polygon = np.asarray(points, dtype=np.float64)
    if len(polygon) < 3:
        return 0.0
    x_values = polygon[:, 0]
    y_values = polygon[:, 1]
    return float(0.5 * abs(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1))))


def centroid(points: np.ndarray) -> np.ndarray:
    """Mean of a set of points."""
    return np.asarray(points, dtype=np.float64)[:, :2].mean(axis=0)


def bounding_box(points: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box (x_min, y_min, x_max, y_max) of points."""
    array = np.asarray(points, dtype=np.float64)
    return float(array[:, 0].min()), float(array[:, 1].min()), float(array[:, 0].max()), float(array[:, 1].max())


def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """Eye aspect ratio from six points ordered outer, upper-1, upper-2, inner, lower-2, lower-1."""
    points = np.asarray(eye_points, dtype=np.float64)
    vertical_first = euclidean_distance(points[1], points[5])
    vertical_second = euclidean_distance(points[2], points[4])
    horizontal = euclidean_distance(points[0], points[3])
    if horizontal < EPSILON:
        return float("nan")
    return (vertical_first + vertical_second) / (2.0 * horizontal)


def path_length(points: np.ndarray) -> float:
    """Total length of a polyline."""
    array = np.asarray(points, dtype=np.float64)
    if len(array) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(array[:, :2], axis=0), axis=1).sum())


def confidence_ellipse_area(points: np.ndarray, chi_square_value: float = 5.991) -> float:
    """Area of the 95% confidence ellipse of 2D points (default chi-square for 2 dof)."""
    array = np.asarray(points, dtype=np.float64)
    if len(array) < 3:
        return 0.0
    covariance = np.cov(array[:, 0], array[:, 1])
    determinant = float(np.linalg.det(covariance))
    if determinant <= 0:
        return 0.0
    return float(math.pi * chi_square_value * math.sqrt(determinant))


def rotate_points(points: np.ndarray, angle_degrees: float, origin: np.ndarray) -> np.ndarray:
    """Rotate points by an angle around an origin."""
    radians = math.radians(angle_degrees)
    rotation = np.array([[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]])
    array = np.asarray(points, dtype=np.float64)[:, :2]
    origin_vector = as_vector(origin)
    return (array - origin_vector) @ rotation.T + origin_vector


def safe_ratio(numerator: float, denominator: float) -> float:
    """Divide, returning NaN when the denominator is effectively zero."""
    if abs(denominator) < EPSILON or math.isnan(denominator):
        return float("nan")
    return numerator / denominator


def asymmetry_ratio(first: float, second: float) -> float:
    """Relative asymmetry |a - b| / max(|a|, |b|) in [0, 1]."""
    larger = max(abs(first), abs(second))
    if larger < EPSILON:
        return 0.0
    return abs(first - second) / larger


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Normalise a vector, returning zeros for a zero vector."""
    array = as_vector(vector)
    norm = float(np.linalg.norm(array))
    if norm < EPSILON:
        return np.zeros(2)
    return array / norm


def fit_line_direction(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a line through points by principal component analysis; returns (centroid, unit direction)."""
    array = np.asarray(points, dtype=np.float64)[:, :2]
    center = array.mean(axis=0)
    centered = array - center
    _, _, right_singular = np.linalg.svd(centered, full_matrices=False)
    direction = right_singular[0]
    return center, unit_vector(direction)
