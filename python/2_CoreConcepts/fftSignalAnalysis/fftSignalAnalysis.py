# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
FFT Signal Analysis

Demonstrates how to analyze signal frequencies using Fast Fourier Transform (FFT):
- Generate composite signals with multiple frequency components
- Use CuPy's cuFFT for GPU-accelerated frequency analysis
- Detect dominant frequencies (peak detection)
- Compare GPU vs CPU FFT performance

Uses cuda.core APIs for device management and timing.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Utilities"))
from cuda_samples_utils import verify_array_result

try:
    import cupy as cp
    import numpy as np
    from cuda.core import Device, EventOptions
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)


def generate_composite_signal(
    num_samples: int,
    sample_rate: float,
    frequencies: list[float],
    amplitudes: list[float],
) -> np.ndarray:
    """
    Generate a composite signal with multiple frequency components.

    Parameters
    ----------
    num_samples : int
        Number of samples in the signal
    sample_rate : float
        Sampling rate in Hz
    frequencies : list[float]
        List of frequency components in Hz
    amplitudes : list[float]
        List of amplitudes for each frequency component

    Returns
    -------
    np.ndarray
        Signal array
    """
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    signal = np.zeros(num_samples, dtype=np.float32)

    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)

    return signal


def find_dominant_frequencies(
    fft_magnitude: cp.ndarray,
    frequencies: cp.ndarray,
    num_peaks: int = 5,
    threshold_ratio: float = 0.1,
) -> list[tuple[float, float]]:
    """
    Find dominant frequencies from FFT magnitude spectrum.

    Uses CPU-based peak detection (transfers magnitude/frequencies via cp.asnumpy).
    Suitable for small-to-medium spectra; for large-scale analysis, consider
    GPU-native peak detection.

    Parameters
    ----------
    fft_magnitude : cp.ndarray
        Magnitude of FFT (positive frequencies only)
    frequencies : cp.ndarray
        Frequency bins
    num_peaks : int
        Maximum number of peaks to return
    threshold_ratio : float
        Minimum peak height as ratio of max peak

    Returns
    -------
    list[tuple[float, float]]
        List of (frequency, magnitude) tuples for detected peaks
    """
    # Find peaks above threshold
    max_magnitude = float(cp.max(fft_magnitude))
    threshold = max_magnitude * threshold_ratio

    # Simple peak detection: find local maxima above threshold
    magnitude_cpu = cp.asnumpy(fft_magnitude)
    freq_cpu = cp.asnumpy(frequencies)

    peaks = []
    for i in range(1, len(magnitude_cpu) - 1):
        if magnitude_cpu[i] > threshold:
            if (
                magnitude_cpu[i] > magnitude_cpu[i - 1]
                and magnitude_cpu[i] > magnitude_cpu[i + 1]
            ):
                peaks.append((freq_cpu[i], magnitude_cpu[i]))

    # Sort by magnitude and return top peaks
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:num_peaks]


def run_fft_analysis(
    num_samples: int = 2**20,
    sample_rate: float = 44100.0,
    device_id: int = 0,
    num_iterations: int = 10,
) -> bool:
    """
    Run FFT signal analysis benchmark.

    device_id and num_iterations are not exposed via CLI; modify defaults
    or call this function directly for customization.

    Parameters
    ----------
    num_samples : int
        Number of samples (power of 2 recommended for FFT)
    sample_rate : float
        Sampling rate in Hz
    device_id : int
        CUDA device ID
    num_iterations : int
        Number of iterations for timing

    Returns
    -------
    bool
        True if analysis succeeded
    """
    print("=" * 60)
    print("FFT Signal Analysis")
    print("=" * 60)

    # Initialize device
    device = Device(device_id)
    device.set_current()
    stream = device.create_stream()

    try:
        print(f"\nDevice: {device.name}")
        print(f"Compute Capability: sm_{device.arch}")

        # Make CuPy use our cuda.core stream
        cp.cuda.ExternalStream(int(stream.handle)).use()

        # Define test signal: composite of multiple frequencies
        test_frequencies = [440.0, 880.0, 1320.0, 2000.0, 5000.0]  # Hz
        test_amplitudes = [1.0, 0.5, 0.3, 0.7, 0.4]

        print("\nSignal Parameters:")
        print(f"  Samples: {num_samples:,}")
        print(f"  Sample Rate: {sample_rate:,.0f} Hz")
        print(f"  Duration: {num_samples / sample_rate:.3f} seconds")
        print(f"  Input Frequencies: {test_frequencies} Hz")
        print(f"  Input Amplitudes: {test_amplitudes}")

        # Generate composite signal on CPU
        h_signal = generate_composite_signal(
            num_samples, sample_rate, test_frequencies, test_amplitudes
        )

        # Transfer to GPU
        d_signal = cp.asarray(h_signal)

        # ---------------------------------------------------------------------
        # GPU FFT (cuFFT via CuPy)
        # ---------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("GPU FFT (cuFFT)")
        print("-" * 60)

        event_opts = EventOptions(enable_timing=True)

        # Warmup
        d_fft_result = cp.fft.rfft(d_signal)
        stream.sync()

        # Timed runs
        start = stream.record(options=event_opts)
        for _ in range(num_iterations):
            d_fft_result = cp.fft.rfft(d_signal)
        end = stream.record(options=event_opts)
        end.sync()

        gpu_time_ms = (end - start) / num_iterations
        print(f"Time: {gpu_time_ms:.3f} ms")

        # Compute magnitude spectrum
        d_magnitude = cp.abs(d_fft_result) * 2 / num_samples
        d_frequencies = cp.fft.rfftfreq(num_samples, 1 / sample_rate)

        # Find dominant frequencies
        detected_peaks = find_dominant_frequencies(d_magnitude, d_frequencies)

        print("\nDetected Frequencies:")
        for freq, mag in detected_peaks:
            print(f"  {freq:8.1f} Hz (magnitude: {mag:.4f})")

        # ---------------------------------------------------------------------
        # CPU FFT (NumPy) for comparison
        # ---------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("CPU FFT (NumPy)")
        print("-" * 60)

        # Warmup
        h_fft_result = np.fft.rfft(h_signal)

        # Timed runs
        cpu_start = time.perf_counter()
        for _ in range(num_iterations):
            h_fft_result = np.fft.rfft(h_signal)
        cpu_end = time.perf_counter()

        cpu_time_ms = (cpu_end - cpu_start) * 1000 / num_iterations
        print(f"Time: {cpu_time_ms:.3f} ms")

        # ---------------------------------------------------------------------
        # Performance Summary
        # ---------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("PERFORMANCE SUMMARY")
        print("-" * 60)
        speedup = cpu_time_ms / gpu_time_ms
        print(f"GPU (cuFFT): {gpu_time_ms:.3f} ms")
        print(f"CPU (NumPy): {cpu_time_ms:.3f} ms")
        print(f"Speedup: {speedup:.1f}x")

        # ---------------------------------------------------------------------
        # Verification
        # ---------------------------------------------------------------------
        print("\n" + "-" * 60)
        print("VERIFICATION")
        print("-" * 60)

        # Compare GPU and CPU results
        h_magnitude = (
            cp.asarray(np.abs(h_fft_result).astype(np.float32)) * 2 / num_samples
        )

        print("GPU vs CPU FFT magnitude: ", end="")
        success = verify_array_result(
            d_magnitude,
            h_magnitude,
            rtol=1e-4,
            atol=1e-6,
        )

        # Verify detected frequencies match input
        print("\nFrequency Detection Accuracy:")
        detected_freqs = [freq for freq, _ in detected_peaks]
        all_found = True
        for expected_freq in test_frequencies:
            found = any(abs(f - expected_freq) < 10 for f in detected_freqs)
            status = "✓" if found else "✗"
            print(f"  {expected_freq:6.0f} Hz: {status}")
            all_found = all_found and found

        success = success and all_found
        return success

    finally:
        # Cleanup - always close resources
        cp.cuda.Stream.null.use()
        stream.close()


def main() -> None:
    """Entry point."""
    success = run_fft_analysis()
    if success:
        print("\nDone")
    else:
        print("\nAnalysis completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
