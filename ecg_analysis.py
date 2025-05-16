import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def generate_ecg(duration=10, fs=250):
    """
    Generate a simulated ECG signal (simple sine wave with peaks).
    duration: seconds
    fs: sampling frequency (Hz)
    """
    t = np.linspace(0, duration, duration * fs)
    # Simulate a basic ECG-like signal with peaks every ~1 second
    ecg_signal = 0.5 * np.sin(2 * np.pi * 1 * t) + 0.05 * np.random.randn(len(t))
    ecg_signal += (np.sin(2 * np.pi * 1 * t) > 0).astype(float) * 0.8  # add sharp peaks
    return t, ecg_signal

def detect_r_peaks(ecg_signal, fs=250):
    """
    Detect R-peaks in the ECG signal using scipy's find_peaks
    """
    peaks, _ = find_peaks(ecg_signal, distance=fs*0.6, height=0.6)
    return peaks

def calculate_hrv(r_peaks, fs=250):
    """
    Calculate basic HRV metrics: RR intervals mean and SDNN (std dev of RR intervals)
    """
    rr_intervals = np.diff(r_peaks) / fs  # convert sample difference to seconds
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    return mean_rr, sdnn

def plot_ecg(t, ecg_signal, r_peaks):
    plt.figure(figsize=(10,4))
    plt.plot(t, ecg_signal, label='ECG Signal')
    plt.plot(t[r_peaks], ecg_signal[r_peaks], 'ro', label='R-peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Simulated ECG Signal with R-Peak Detection')
    plt.legend()
    plt.show()

def main():
    fs = 250
    t, ecg_signal = generate_ecg(duration=10, fs=fs)
    r_peaks = detect_r_peaks(ecg_signal, fs)
    mean_rr, sdnn = calculate_hrv(r_peaks, fs)

    print(f"Mean RR interval: {mean_rr:.3f} seconds")
    print(f"SDNN (HRV measure): {sdnn:.3f} seconds")

    plot_ecg(t, ecg_signal, r_peaks)

if __name__ == "__main__":
    main()
