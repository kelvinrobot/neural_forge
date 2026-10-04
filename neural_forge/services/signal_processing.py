"""Signal processing on irregularly sampled landmark time series using real timestamps."""

from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats
from scipy.integrate import trapezoid

EPSILON = 1e-12


@dataclass(frozen=True)
class UniformSignal:
    """A signal resampled to a uniform time grid."""

    timestamps: np.ndarray
    values: np.ndarray
    sampling_rate: float

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


@dataclass(frozen=True)
class SpectralPeak:
    """Dominant spectral peak within a band."""

    frequency_hz: float
    peak_power: float
    band_power: float
    regularity: float


@dataclass(frozen=True)
class TrendResult:
    """Linear trend statistics."""

    slope: float
    intercept: float
    r_value: float


def estimate_sampling_rate(timestamps: np.ndarray) -> float:
    """Median sampling rate in Hz computed from measured timestamps."""
    times = np.asarray(timestamps, dtype=np.float64)
    if len(times) < 2:
        return 0.0
    intervals = np.diff(np.sort(times))
    intervals = intervals[intervals > EPSILON]
    if len(intervals) == 0:
        return 0.0
    return float(1.0 / np.median(intervals))


def resample_uniform(
    timestamps: np.ndarray,
    values: np.ndarray,
    sampling_rate: float | None = None,
    max_sampling_rate: float = 60.0,
) -> UniformSignal:
    """Linearly interpolate samples onto a uniform grid at the measured (or given) rate."""
    times = np.asarray(timestamps, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(times)
    times = times[order]
    data = data[order]
    unique_mask = np.concatenate(([True], np.diff(times) > EPSILON))
    times = times[unique_mask]
    data = data[unique_mask]
    if len(times) < 2:
        return UniformSignal(timestamps=times, values=data, sampling_rate=0.0)
    rate = sampling_rate if sampling_rate is not None else estimate_sampling_rate(times)
    rate = float(min(max(rate, 1.0), max_sampling_rate))
    grid = np.arange(times[0], times[-1], 1.0 / rate)
    if len(grid) < 2:
        return UniformSignal(timestamps=times, values=data, sampling_rate=rate)
    if data.ndim == 1:
        resampled = np.interp(grid, times, data)
    else:
        resampled = np.stack([np.interp(grid, times, data[:, column]) for column in range(data.shape[1])], axis=1)
    return UniformSignal(timestamps=grid, values=resampled, sampling_rate=rate)


def detrend_signal(values: np.ndarray, kind: str = "linear") -> np.ndarray:
    """Remove a linear or constant trend."""
    data = np.asarray(values, dtype=np.float64)
    if len(data) < 3:
        return data - data.mean(axis=0) if len(data) else data
    return scipy_signal.detrend(data, axis=0, type=kind)


def butter_bandpass(values: np.ndarray, sampling_rate: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter with Nyquist-aware limits."""
    data = detrend_signal(values)
    if sampling_rate <= 0 or len(data) < 4:
        return data
    nyquist = sampling_rate / 2.0
    effective_high = min(high_hz, 0.95 * nyquist)
    effective_low = max(low_hz, 0.01)
    if effective_low >= effective_high:
        return data
    sos = scipy_signal.butter(order, [effective_low / nyquist, effective_high / nyquist], btype="band", output="sos")
    minimum_length = 3 * (2 * sos.shape[0] + 1)
    if len(data) <= minimum_length:
        return data
    return scipy_signal.sosfiltfilt(sos, data, axis=0)


def butter_lowpass(values: np.ndarray, sampling_rate: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter."""
    data = np.asarray(values, dtype=np.float64)
    if sampling_rate <= 0 or len(data) < 4:
        return data
    nyquist = sampling_rate / 2.0
    effective_cutoff = min(cutoff_hz, 0.95 * nyquist)
    if effective_cutoff <= 0:
        return data
    sos = scipy_signal.butter(order, effective_cutoff / nyquist, btype="low", output="sos")
    minimum_length = 3 * (2 * sos.shape[0] + 1)
    if len(data) <= minimum_length:
        return data
    return scipy_signal.sosfiltfilt(sos, data, axis=0)


def butter_highpass(values: np.ndarray, sampling_rate: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter."""
    data = np.asarray(values, dtype=np.float64)
    if sampling_rate <= 0 or len(data) < 4:
        return data - data.mean(axis=0) if len(data) else data
    nyquist = sampling_rate / 2.0
    effective_cutoff = min(cutoff_hz, 0.95 * nyquist)
    if effective_cutoff <= 0:
        return data
    sos = scipy_signal.butter(order, effective_cutoff / nyquist, btype="high", output="sos")
    minimum_length = 3 * (2 * sos.shape[0] + 1)
    if len(data) <= minimum_length:
        return data - data.mean(axis=0)
    return scipy_signal.sosfiltfilt(sos, data, axis=0)


def welch_spectrum(values: np.ndarray, sampling_rate: float, segment_seconds: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density estimate."""
    data = np.asarray(values, dtype=np.float64)
    if len(data) < 8 or sampling_rate <= 0:
        return np.zeros(0), np.zeros(0)
    segment_length = int(min(len(data), max(8, round(sampling_rate * segment_seconds))))
    frequencies, power = scipy_signal.welch(data, fs=sampling_rate, nperseg=segment_length, detrend="linear")
    return frequencies, power


def fft_spectrum(values: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided FFT amplitude spectrum with a Hann window."""
    data = detrend_signal(np.asarray(values, dtype=np.float64))
    if len(data) < 4 or sampling_rate <= 0:
        return np.zeros(0), np.zeros(0)
    window = np.hanning(len(data))
    amplitudes = np.abs(np.fft.rfft(data * window)) * 2.0 / max(window.sum(), EPSILON)
    frequencies = np.fft.rfftfreq(len(data), d=1.0 / sampling_rate)
    return frequencies, amplitudes


def band_power(frequencies: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float) -> float:
    """Integrated power within a frequency band."""
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if mask.sum() < 2:
        return float(power[mask].sum()) if mask.any() else 0.0
    return float(trapezoid(power[mask], frequencies[mask]))


def dominant_peak(
    frequencies: np.ndarray,
    power: np.ndarray,
    low_hz: float,
    high_hz: float,
    peak_half_width_hz: float = 0.5,
) -> SpectralPeak | None:
    """Dominant frequency in a band with a regularity score (fraction of band power near the peak)."""
    if len(frequencies) == 0:
        return None
    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not band_mask.any():
        return None
    band_frequencies = frequencies[band_mask]
    band_spectrum = power[band_mask]
    peak_index = int(np.argmax(band_spectrum))
    peak_frequency = float(band_frequencies[peak_index])
    total = band_power(frequencies, power, low_hz, high_hz)
    near_peak = band_power(
        frequencies, power, max(low_hz, peak_frequency - peak_half_width_hz), min(high_hz, peak_frequency + peak_half_width_hz)
    )
    regularity = float(near_peak / total) if total > EPSILON else 0.0
    return SpectralPeak(
        frequency_hz=peak_frequency,
        peak_power=float(band_spectrum[peak_index]),
        band_power=total,
        regularity=float(np.clip(regularity, 0.0, 1.0)),
    )


def spectral_flatness(power: np.ndarray) -> float:
    """Wiener entropy: geometric over arithmetic mean of the spectrum (1 = noise-like, 0 = tonal)."""
    spectrum = np.asarray(power, dtype=np.float64)
    spectrum = spectrum[spectrum > EPSILON]
    if len(spectrum) == 0:
        return 0.0
    geometric_mean = float(np.exp(np.mean(np.log(spectrum))))
    arithmetic_mean = float(np.mean(spectrum))
    return geometric_mean / arithmetic_mean if arithmetic_mean > EPSILON else 0.0


def find_signal_peaks(values: np.ndarray, sampling_rate: float, min_interval_seconds: float, prominence: float) -> np.ndarray:
    """Indices of peaks separated by a minimum interval and with a minimum prominence."""
    data = np.asarray(values, dtype=np.float64)
    if len(data) < 3:
        return np.zeros(0, dtype=int)
    distance = max(1, int(round(min_interval_seconds * sampling_rate))) if sampling_rate > 0 else 1
    peaks, _ = scipy_signal.find_peaks(data, distance=distance, prominence=max(prominence, EPSILON))
    return peaks


def find_peaks_irregular(
    timestamps: np.ndarray, values: np.ndarray, min_interval_seconds: float, prominence: float
) -> np.ndarray:
    """Peak indices on an irregular series, enforcing a minimum time between peaks."""
    data = np.asarray(values, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if len(data) < 3:
        return np.zeros(0, dtype=int)
    candidates, properties = scipy_signal.find_peaks(data, prominence=max(prominence, EPSILON))
    if len(candidates) == 0:
        return candidates
    prominences = properties["prominences"]
    order = np.argsort(prominences)[::-1]
    selected: list[int] = []
    for candidate_position in order:
        candidate = int(candidates[candidate_position])
        if all(abs(times[candidate] - times[chosen]) >= min_interval_seconds for chosen in selected):
            selected.append(candidate)
    return np.array(sorted(selected), dtype=int)


def root_mean_square(values: np.ndarray) -> float:
    """Root mean square of a signal."""
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data))))


def linear_trend(x_values: np.ndarray, y_values: np.ndarray) -> TrendResult:
    """Least-squares linear trend of y against x."""
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    if len(x_array) < 2 or float(np.ptp(x_array)) < EPSILON:
        return TrendResult(slope=0.0, intercept=float(y_array.mean()) if len(y_array) else 0.0, r_value=0.0)
    result = scipy_stats.linregress(x_array, y_array)
    return TrendResult(slope=float(result.slope), intercept=float(result.intercept), r_value=float(result.rvalue))


def coefficient_of_variation(values: np.ndarray) -> float:
    """Standard deviation divided by mean."""
    data = np.asarray(values, dtype=np.float64)
    if len(data) < 2:
        return 0.0
    mean_value = float(np.mean(data))
    if abs(mean_value) < EPSILON:
        return 0.0
    return float(np.std(data, ddof=1) / abs(mean_value))


def time_weighted_fraction(timestamps: np.ndarray, condition: np.ndarray) -> float:
    """Fraction of elapsed time during which a boolean condition holds, using real timestamps."""
    times = np.asarray(timestamps, dtype=np.float64)
    flags = np.asarray(condition, dtype=bool)
    if len(times) < 2:
        return float(flags.mean()) if len(flags) else 0.0
    durations = np.diff(times)
    total = float(durations.sum())
    if total <= EPSILON:
        return float(flags.mean())
    return float(durations[flags[:-1]].sum() / total)


def contiguous_segments(timestamps: np.ndarray, condition: np.ndarray) -> list[tuple[float, float, int, int]]:
    """Runs where a condition holds as (start_time, end_time, start_index, end_index)."""
    flags = np.asarray(condition, dtype=bool)
    times = np.asarray(timestamps, dtype=np.float64)
    segments: list[tuple[float, float, int, int]] = []
    start_index: int | None = None
    for index, flag in enumerate(flags):
        if flag and start_index is None:
            start_index = index
        elif not flag and start_index is not None:
            segments.append((float(times[start_index]), float(times[index]), start_index, index - 1))
            start_index = None
    if start_index is not None:
        segments.append((float(times[start_index]), float(times[-1]), start_index, len(flags) - 1))
    return segments


def velocity(timestamps: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Finite-difference velocity using real timestamps; positions may be 1D or (N, D)."""
    times = np.asarray(timestamps, dtype=np.float64)
    data = np.asarray(positions, dtype=np.float64)
    if len(times) < 2:
        return np.zeros_like(data)
    return np.gradient(data, times, axis=0)


def skewness(values: np.ndarray) -> float:
    """Sample skewness of a signal."""
    data = np.asarray(values, dtype=np.float64)
    if len(data) < 3 or float(np.std(data)) < EPSILON:
        return 0.0
    return float(scipy_stats.skew(data))
