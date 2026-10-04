"""Tests for colour conversion, white balance and CIELAB statistics."""

import numpy as np

from neural_forge.services import colour_utils as cu
from tests.synthetic import SKIN_BGR


def test_bgr_to_lab_white_and_black() -> None:
    white = cu.bgr_to_lab(np.full((4, 4, 3), 255, dtype=np.uint8))
    black = cu.bgr_to_lab(np.zeros((4, 4, 3), dtype=np.uint8))
    assert abs(float(white[0, 0, 0]) - 100.0) < 0.5
    assert abs(float(black[0, 0, 0])) < 0.5


def test_grey_world_gains_neutral_for_grey_image() -> None:
    gains = cu.grey_world_gains(np.full((50, 50, 3), 128, dtype=np.uint8), 20, 235)
    np.testing.assert_allclose(gains, np.ones(3), atol=1e-6)


def test_grey_world_corrects_colour_cast() -> None:
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    image[:, :] = (100, 120, 160)
    gains = cu.grey_world_gains(image, 20, 235)
    corrected = cu.apply_channel_gains(image, gains).astype(float)
    assert np.ptp(corrected[0, 0]) < 2.0


def test_masked_lab_statistics_and_difference() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :10] = (40, 40, 200)
    image[:, 10:] = SKIN_BGR
    lab = cu.bgr_to_lab(image)
    left_mask = np.zeros((20, 20), dtype=np.uint8)
    left_mask[:, :10] = 255
    right_mask = 255 - left_mask
    red = cu.masked_lab_statistics(lab, left_mask)
    skin = cu.masked_lab_statistics(lab, right_mask)
    assert red is not None and skin is not None
    difference = cu.region_difference(red, skin)
    assert difference.delta_red_green > 0
    assert cu.masked_lab_statistics(lab, np.zeros((20, 20), dtype=np.uint8), 5) is None


def test_skin_mask_detects_skin_colour() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, :20] = SKIN_BGR
    image[:, 20:] = (200, 50, 50)
    mask = cu.skin_mask_ycrcb(image, (133, 180), (77, 135))
    assert mask[:, :15].mean() > 200
    assert mask[:, 25:].mean() < 50


def test_crop_and_expand_box() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = cu.crop_image(image, cu.expand_box((40, 40, 60, 60), 0.5))
    assert crop.shape[:2] == (40, 40)
    assert cu.clip_box((-10, -10, 200, 200), 100, 100) == (0, 0, 100, 100)
    assert cu.encode_jpeg_base64(image)
