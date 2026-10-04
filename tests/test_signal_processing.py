"""Tests for signal processing helpers using irregular real-style timestamps."""

import numpy as np

from neural_forge.services import signal_processing as sp


def irregular_timestamps(duration: float, rate: float, jitter: float = 0.2, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    intervals = (1.0 / rate) * (1.0 + jitter * (generator.random(int(duration * rate)) - 0.5))
    return np.cumsum(intervals)


def test_estimate_sampling_rate_uses_measured_intervals() -> None:
    timestamps = irregular_timestamps(10.0, 25.0)
    assert abs(sp.estimate_sampling_rate(timestamps) - 25.0) < 2.0


def test_resample_uniform_preserves_linear_signal() -> None:
    timestamps = irregular_timestamps(5.0, 20.0)
    values = 2.0 * timestamps + 1.0
    uniform = sp.resample_uniform(timestamps, values)
    assert uniform.sampling_rate > 15.0
    np.testing.assert_allclose(uniform.values, 2.0 * uniform.timestamps + 1.0, atol=1e-6)


def test_bandpass_and_welch_find_tremor_frequency() -> None:
    timestamps = irregular_timestamps(10.0, 30.0)
    values = 0.5 * np.sin(2 * np.pi * 5.0 * timestamps) + 0.3 * timestamps
    uniform = sp.resample_uniform(timestamps, values)
    filtered = sp.butter_bandpass(uniform.values, uniform.sampling_rate, 3.0, 12.0)
    frequencies, power = sp.welch_spectrum(filtered, uniform.sampling_rate)
    peak = sp.dominant_peak(frequencies, power, 3.0, 12.0)
    assert peak is not None
    assert abs(peak.frequency_hz - 5.0) < 0.5
    assert peak.regularity > 0.5


def test_bandpass_respects_nyquist_limit() -> None:
    timestamps = np.arange(0, 5, 1 / 10.0)
    values = np.sin(2 * np.pi * 2.0 * timestamps)
    filtered = sp.butter_bandpass(values, 10.0, 1.0, 20.0)
    assert filtered.shape == values.shape
    assert np.all(np.isfinite(filtered))


def test_find_peaks_irregular_counts_taps() -> None:
    timestamps = irregular_timestamps(5.0, 30.0)
    values = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * timestamps)
    peaks = sp.find_peaks_irregular(timestamps, values, 0.1, 0.3)
    assert 13 <= len(peaks) <= 16


def test_time_weighted_fraction_uses_durations() -> None:
    timestamps = np.array([0.0, 1.0, 1.1, 3.0])
    condition = np.array([True, False, True, False])
    assert abs(sp.time_weighted_fraction(timestamps, condition) - (1.0 + 1.9) / 3.0) < 1e-9


def test_linear_trend_slope() -> None:
    x_values = np.arange(10, dtype=float)
    trend = sp.linear_trend(x_values, 3.0 * x_values + 2.0)
    assert abs(trend.slope - 3.0) < 1e-9
    assert abs(trend.r_value - 1.0) < 1e-9


def test_spectral_flatness_distinguishes_noise_from_tone() -> None:
    generator = np.random.default_rng(1)
    noise = generator.normal(size=2048)
    tone = np.sin(2 * np.pi * 50 * np.arange(2048) / 1000.0)
    _, noise_power = sp.welch_spectrum(noise, 1000.0, segment_seconds=0.256)
    _, tone_power = sp.welch_spectrum(tone, 1000.0, segment_seconds=0.256)
    assert sp.spectral_flatness(noise_power) > sp.spectral_flatness(tone_power)


def test_contiguous_segments() -> None:
    timestamps = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    segments = sp.contiguous_segments(timestamps, np.array([False, True, True, False, True]))
    assert segments[0][:2] == (0.5, 1.5)
    assert segments[-1][0] == 2.0
