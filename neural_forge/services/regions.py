"""Anatomical landmark indices and region masks for face, hand and body measurements.

Side names always refer to the patient's anatomical side on unmirrored frames.
"""

import cv2
import numpy as np

from neural_forge.models.mediapipe_face import FaceLandmarks
from neural_forge.models.mediapipe_hands import HandLandmarks
from neural_forge.services.geometry import euclidean_distance, fit_line_direction, midpoint, unit_vector

FACE_FOREHEAD_TOP = 10
FACE_CHIN = 152
NOSE_TIP = 1
NOSE_BRIDGE = 168
FACIAL_MIDLINE_UPPER = (10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1)

RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_EYE_UPPER = 159
RIGHT_EYE_LOWER = 145
LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
LEFT_EYE_UPPER = 386
LEFT_EYE_LOWER = 374
RIGHT_EYE_EAR_POINTS = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR_POINTS = (362, 385, 387, 263, 373, 380)
RIGHT_UPPER_LID = (33, 246, 161, 160, 159, 158, 157, 173, 133)
RIGHT_LOWER_LID = (33, 7, 163, 144, 145, 153, 154, 155, 133)
LEFT_UPPER_LID = (263, 466, 388, 387, 386, 385, 384, 398, 362)
LEFT_LOWER_LID = (263, 249, 390, 373, 374, 380, 381, 382, 362)
RIGHT_IRIS_CENTER = 468
RIGHT_IRIS_RING = (469, 470, 471, 472)
LEFT_IRIS_CENTER = 473
LEFT_IRIS_RING = (474, 475, 476, 477)

RIGHT_BROW = (70, 63, 105, 66, 107)
LEFT_BROW = (300, 293, 334, 296, 336)
RIGHT_BROW_MID = 105
LEFT_BROW_MID = 334

MOUTH_RIGHT_CORNER = 61
MOUTH_LEFT_CORNER = 291
UPPER_LIP_INNER = 13
LOWER_LIP_INNER = 14
UPPER_LIP_OUTER = 0
LOWER_LIP_OUTER = 17
LIPS_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
LIPS_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)

RIGHT_ALAR_BASE = 98
LEFT_ALAR_BASE = 327
RIGHT_FACE_CONTOUR = 234
LEFT_FACE_CONTOUR = 454
RIGHT_CHEEK_REGION = (117, 118, 101, 36, 205, 187, 123, 116)
LEFT_CHEEK_REGION = (346, 347, 330, 266, 425, 411, 352, 345)
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)

EYE_LANDMARKS: dict[str, dict[str, object]] = {
    "right": {
        "outer": RIGHT_EYE_OUTER,
        "inner": RIGHT_EYE_INNER,
        "upper": RIGHT_EYE_UPPER,
        "lower": RIGHT_EYE_LOWER,
        "ear_points": RIGHT_EYE_EAR_POINTS,
        "upper_lid": RIGHT_UPPER_LID,
        "lower_lid": RIGHT_LOWER_LID,
        "iris_center": RIGHT_IRIS_CENTER,
        "iris_ring": RIGHT_IRIS_RING,
        "brow": RIGHT_BROW,
        "brow_mid": RIGHT_BROW_MID,
        "cheek_region": RIGHT_CHEEK_REGION,
        "mouth_corner": MOUTH_RIGHT_CORNER,
        "alar_base": RIGHT_ALAR_BASE,
        "face_contour": RIGHT_FACE_CONTOUR,
    },
    "left": {
        "outer": LEFT_EYE_OUTER,
        "inner": LEFT_EYE_INNER,
        "upper": LEFT_EYE_UPPER,
        "lower": LEFT_EYE_LOWER,
        "ear_points": LEFT_EYE_EAR_POINTS,
        "upper_lid": LEFT_UPPER_LID,
        "lower_lid": LEFT_LOWER_LID,
        "iris_center": LEFT_IRIS_CENTER,
        "iris_ring": LEFT_IRIS_RING,
        "brow": LEFT_BROW,
        "brow_mid": LEFT_BROW_MID,
        "cheek_region": LEFT_CHEEK_REGION,
        "mouth_corner": MOUTH_LEFT_CORNER,
        "alar_base": LEFT_ALAR_BASE,
        "face_contour": LEFT_FACE_CONTOUR,
    },
}

HAND_WRIST = 0
FINGER_CHAINS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}
PALM_POLYGON = (0, 1, 5, 9, 13, 17)


def landmark_index(side: str, key: str) -> int:
    """Return a single landmark index for an anatomical side and key."""
    return int(EYE_LANDMARKS[side][key])  # type: ignore[arg-type]


def landmark_indices(side: str, key: str) -> tuple[int, ...]:
    """Return a tuple of landmark indices for an anatomical side and key."""
    return tuple(EYE_LANDMARKS[side][key])  # type: ignore[arg-type]


def polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    """Filled polygon mask (0/255) of the given image shape."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
    return mask


def circle_mask(shape: tuple[int, int], center: np.ndarray, radius: float) -> np.ndarray:
    """Filled circle mask (0/255)."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if radius > 0:
        cv2.circle(mask, (int(round(center[0])), int(round(center[1]))), int(round(radius)), 255, -1)
    return mask


def eye_centre(face: FaceLandmarks, side: str) -> np.ndarray:
    """Midpoint of the eye corners."""
    return midpoint(face.point(landmark_index(side, "outer")), face.point(landmark_index(side, "inner")))


def inter_ocular_distance(face: FaceLandmarks) -> float:
    """Distance between the two eye centres in pixels."""
    return euclidean_distance(eye_centre(face, "right"), eye_centre(face, "left"))


def eye_width(face: FaceLandmarks, side: str) -> float:
    """Distance between the eye corners in pixels."""
    return euclidean_distance(face.point(landmark_index(side, "outer")), face.point(landmark_index(side, "inner")))


def iris_geometry(face: FaceLandmarks, side: str) -> tuple[np.ndarray, float]:
    """Iris centre and radius in pixels."""
    center = face.point(landmark_index(side, "iris_center"))
    ring = face.points(landmark_indices(side, "iris_ring"))
    horizontal_diameter = euclidean_distance(ring[0], ring[2])
    vertical_diameter = euclidean_distance(ring[1], ring[3])
    return center, (horizontal_diameter + vertical_diameter) / 4.0


def eye_polygon(face: FaceLandmarks, side: str) -> np.ndarray:
    """Closed polygon of the visible eye opening."""
    upper = face.points(landmark_indices(side, "upper_lid"))
    lower = face.points(landmark_indices(side, "lower_lid"))
    return np.concatenate([upper, lower[::-1][1:-1]], axis=0)


def sclera_mask(face: FaceLandmarks, side: str, shape: tuple[int, int], iris_margin: float = 1.1) -> np.ndarray:
    """Visible eye opening minus the iris disc."""
    opening = polygon_mask(shape, eye_polygon(face, side))
    center, radius = iris_geometry(face, side)
    iris = circle_mask(shape, center, radius * iris_margin)
    return cv2.bitwise_and(opening, cv2.bitwise_not(iris))


def lower_conjunctiva_mask(face: FaceLandmarks, side: str, shape: tuple[int, int], band_fraction: float) -> np.ndarray:
    """Band below the lower lid margin where the palpebral conjunctiva is exposed when the lid is pulled down."""
    lower = face.points(landmark_indices(side, "lower_lid"))[1:-1]
    band_height = band_fraction * eye_width(face, side)
    shifted = lower + np.array([0.0, band_height])
    polygon = np.concatenate([lower, shifted[::-1]], axis=0)
    return polygon_mask(shape, polygon)


def lips_mask(face: FaceLandmarks, shape: tuple[int, int]) -> np.ndarray:
    """Lip vermilion: outer lip contour minus the inner mouth opening."""
    outer = polygon_mask(shape, face.points(LIPS_OUTER))
    inner = polygon_mask(shape, face.points(LIPS_INNER))
    return cv2.bitwise_and(outer, cv2.bitwise_not(inner))


def mouth_interior_mask(face: FaceLandmarks, shape: tuple[int, int]) -> np.ndarray:
    """Inner mouth opening."""
    return polygon_mask(shape, face.points(LIPS_INNER))


def cheek_mask(face: FaceLandmarks, side: str, shape: tuple[int, int]) -> np.ndarray:
    """Cheek skin region used as the patient's own reference skin."""
    return polygon_mask(shape, face.points(landmark_indices(side, "cheek_region")))


def forehead_mask(face: FaceLandmarks, shape: tuple[int, int]) -> np.ndarray:
    """Rectangular forehead region between the eyebrows and the upper face boundary."""
    right_brow = face.point(RIGHT_BROW_MID)
    left_brow = face.point(LEFT_BROW_MID)
    top_point = face.point(FACE_FOREHEAD_TOP)
    brow_level = (right_brow[1] + left_brow[1]) / 2.0
    distance = inter_ocular_distance(face)
    bottom = brow_level - 0.12 * distance
    top = top_point[1] + 0.3 * (brow_level - top_point[1])
    x_left = min(right_brow[0], left_brow[0])
    x_right = max(right_brow[0], left_brow[0])
    if bottom <= top:
        return np.zeros(shape[:2], dtype=np.uint8)
    polygon = np.array([[x_left, top], [x_right, top], [x_right, bottom], [x_left, bottom]])
    return polygon_mask(shape, polygon)


def reference_skin_mask(face: FaceLandmarks, shape: tuple[int, int]) -> np.ndarray:
    """Combined cheek and forehead skin reference mask."""
    combined = cv2.bitwise_or(cheek_mask(face, "right", shape), cheek_mask(face, "left", shape))
    return cv2.bitwise_or(combined, forehead_mask(face, shape))


def eye_crop_box(face: FaceLandmarks, side: str, padding_fraction: float) -> tuple[float, float, float, float]:
    """Bounding box around one eye including lids and brow margin."""
    points = np.concatenate([eye_polygon(face, side), face.points(landmark_indices(side, "brow"))], axis=0)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    return (
        x_min - width * padding_fraction,
        y_min - height * padding_fraction,
        x_max + width * padding_fraction,
        y_max + height * padding_fraction,
    )


def both_eyes_crop_box(face: FaceLandmarks, padding_fraction: float) -> tuple[float, float, float, float]:
    """Bounding box covering both eyes and brows."""
    right_box = eye_crop_box(face, "right", padding_fraction)
    left_box = eye_crop_box(face, "left", padding_fraction)
    return (
        min(right_box[0], left_box[0]),
        min(right_box[1], left_box[1]),
        max(right_box[2], left_box[2]),
        max(right_box[3], left_box[3]),
    )


def mouth_crop_box(face: FaceLandmarks, padding_fraction: float) -> tuple[float, float, float, float]:
    """Bounding box around the lips."""
    points = face.points(LIPS_OUTER)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    return (
        x_min - width * padding_fraction,
        y_min - height * padding_fraction,
        x_max + width * padding_fraction,
        y_max + height * padding_fraction,
    )


def mouth_corner_box(face: FaceLandmarks, side: str, size_fraction: float) -> tuple[float, float, float, float]:
    """Square box centred on one mouth corner sized relative to mouth width."""
    corner = face.point(landmark_index(side, "mouth_corner"))
    mouth_width = euclidean_distance(face.point(MOUTH_RIGHT_CORNER), face.point(MOUTH_LEFT_CORNER))
    half_size = size_fraction * mouth_width / 2.0
    return corner[0] - half_size, corner[1] - half_size, corner[0] + half_size, corner[1] + half_size


def mouth_width(face: FaceLandmarks) -> float:
    """Distance between the mouth corners."""
    return euclidean_distance(face.point(MOUTH_RIGHT_CORNER), face.point(MOUTH_LEFT_CORNER))


def facial_midline(face: FaceLandmarks) -> tuple[np.ndarray, np.ndarray]:
    """Facial midline (point, unit direction pointing down the face) fitted to upper-face points."""
    center, direction = fit_line_direction(face.points(FACIAL_MIDLINE_UPPER))
    downward = face.point(NOSE_TIP) - face.point(FACE_FOREHEAD_TOP)
    if float(np.dot(direction, downward)) < 0:
        direction = -direction
    return center, direction


def face_aligned_pixels(face: FaceLandmarks) -> np.ndarray:
    """All landmarks rotated about the nose bridge so the eye line is horizontal; +x points to the patient's left."""
    eye_vector = eye_centre(face, "left") - eye_centre(face, "right")
    roll = float(np.arctan2(eye_vector[1], eye_vector[0]))
    cosine = float(np.cos(-roll))
    sine = float(np.sin(-roll))
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    origin = face.point(NOSE_BRIDGE)
    return (face.pixels[:, :2] - origin) @ rotation.T


def patient_left_direction(face: FaceLandmarks) -> np.ndarray:
    """Unit vector in the image pointing toward the patient's left side."""
    return unit_vector(eye_centre(face, "left") - eye_centre(face, "right"))


def palm_mask(hand: HandLandmarks, shape: tuple[int, int], erosion_fraction: float = 0.08) -> np.ndarray:
    """Palm or back-of-hand region bounded by wrist and knuckles, eroded away from edges."""
    mask = polygon_mask(shape, hand.pixels[list(PALM_POLYGON)])
    kernel_size = max(3, int(hand.palm_length() * erosion_fraction) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask, kernel)


def nail_bed_circle(hand: HandLandmarks, finger: str, radius_fraction: float) -> tuple[np.ndarray, float]:
    """Centre and radius of the nail-bed region of a finger."""
    chain = FINGER_CHAINS[finger]
    tip = hand.point(chain[3])
    distal_joint = hand.point(chain[2])
    center = tip + (distal_joint - tip) * 0.3
    radius = radius_fraction * euclidean_distance(tip, distal_joint)
    return center, radius


def nail_bed_mask(hand: HandLandmarks, finger: str, shape: tuple[int, int], radius_fraction: float) -> np.ndarray:
    """Mask of the nail-bed region of a finger."""
    center, radius = nail_bed_circle(hand, finger, radius_fraction)
    return circle_mask(shape, center, radius)


def forearm_patch_mask(hand: HandLandmarks, shape: tuple[int, int], length_fraction: float, width_fraction: float) -> np.ndarray:
    """Distal forearm skin patch just beyond the wrist, opposite the fingers."""
    wrist = hand.point(HAND_WRIST)
    knuckle = hand.point(9)
    palm_length = euclidean_distance(wrist, knuckle)
    along = unit_vector(wrist - knuckle)
    across = np.array([-along[1], along[0]])
    start = wrist + along * palm_length * 0.15
    end = wrist + along * palm_length * (0.15 + length_fraction)
    half_width = palm_length * width_fraction / 2.0
    polygon = np.array([start + across * half_width, end + across * half_width, end - across * half_width, start - across * half_width])
    return polygon_mask(shape, polygon)


def finger_box(hand: HandLandmarks, finger: str, padding_fraction: float) -> tuple[float, float, float, float]:
    """Bounding box of a finger's landmarks with padding."""
    chain = FINGER_CHAINS[finger]
    points = hand.pixels[list(chain)]
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    size = max(x_max - x_min, y_max - y_min, 1.0)
    return (
        x_min - size * padding_fraction,
        y_min - size * padding_fraction,
        x_max + size * padding_fraction,
        y_max + size * padding_fraction,
    )


def hand_box(hand: HandLandmarks, padding_fraction: float) -> tuple[float, float, float, float]:
    """Bounding box of the hand with padding."""
    x_min, y_min, x_max, y_max = hand.bounding_box()
    width = x_max - x_min
    height = y_max - y_min
    return (
        x_min - width * padding_fraction,
        y_min - height * padding_fraction,
        x_max + width * padding_fraction,
        y_max + height * padding_fraction,
    )


def width_profile(mask: np.ndarray, center: np.ndarray, axis_direction: np.ndarray, max_half_width: float) -> float:
    """Width of the contiguous foreground run through a point, measured perpendicular to an axis."""
    across = np.array([-axis_direction[1], axis_direction[0]])
    height, width = mask.shape[:2]
    steps = int(max(2, max_half_width))

    def run_length(direction_sign: float) -> int:
        length = 0
        for step in range(1, steps + 1):
            sample = center + across * step * direction_sign
            x_coordinate = int(round(sample[0]))
            y_coordinate = int(round(sample[1]))
            if x_coordinate < 0 or y_coordinate < 0 or x_coordinate >= width or y_coordinate >= height:
                break
            if mask[y_coordinate, x_coordinate] == 0:
                break
            length += 1
        return length

    x_center = int(round(center[0]))
    y_center = int(round(center[1]))
    if x_center < 0 or y_center < 0 or x_center >= width or y_center >= height or mask[y_center, x_center] == 0:
        return 0.0
    return float(run_length(1.0) + run_length(-1.0) + 1)
