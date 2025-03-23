import sklearn.preprocessing as sp
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import neurokit2 as nk
import matplotlib.pyplot as plt
def compare_fixed_vs_adaptive(data, rpeaks, sampling_rate, final_length=300, onset=100, offset=200):
    rpeaks_list = rpeaks['ECG_R_Peaks']
    fixed_beats = []
    adaptive_beats = []

    for i in range(1, len(rpeaks_list) - 1):
        r = rpeaks_list[i]
        prev_rr = r - rpeaks_list[i - 1]
        next_rr = rpeaks_list[i + 1] - r

        # --- Fixed window ---
        start_f = r - onset
        end_f = r + offset
        if start_f >= 0 and end_f < len(data):
            beat_fixed = data[start_f:end_f]
            beat_fixed = nk.signal.signal_resample(beat_fixed, desired_length=final_length,sampling_rate=sampling_rate)
            fixed_beats.append(beat_fixed)

        # --- Adaptive window ---
        total_window = int(0.5*(prev_rr + next_rr))  # Or however much you want
        half_window = total_window // 2

        start_a = r - half_window
        end_a = r + half_window
        if start_a >= 0 and end_a < len(data):
            beat_adaptive = data[start_a:end_a]
            beat_adaptive = nk.signal.signal_resample(beat_adaptive, desired_length=final_length,sampling_rate=sampling_rate)
            adaptive_beats.append(beat_adaptive)

    # --- Plot comparison ---
    num_to_plot = min(10, len(fixed_beats), len(adaptive_beats))
    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    for i in range(num_to_plot):
        axs[0].plot(fixed_beats[i], alpha=0.5, label=f"Beat {i+1}")
        axs[1].plot(adaptive_beats[i], alpha=0.5, label=f"Beat {i+1}")

    axs[0].set_title("Fixed Windowed Beats")
    axs[1].set_title("RR-Adaptive Windowed Beats")

    for ax in axs:
        ax.grid(True)
        ax.set_ylabel("Amplitude")
    axs[-1].set_xlabel("Resampled Samples")
    plt.tight_layout()
    plt.show()

class Preprocess:
    def __init__(self, onset: int, offset: int, final_length: int = None, peak='R'):
        self.onset = onset
        self.offset = offset
        self.window_length = onset + offset
        self.final_length = final_length or self.window_length
        self.peak = peak


    def clean(self, ecg_signal, sampling_rate):
        """Resample ECG to 500Hz and apply Neurokit2's ECG cleaning."""
        print("\n--- Debug: ECG Signal Before Cleaning ---")
        print("First 10 values:", ecg_signal[:10] if len(ecg_signal) > 10 else ecg_signal)
        print("Min:", np.min(ecg_signal), "Max:", np.max(ecg_signal))
        print("Mean:", np.mean(ecg_signal), "Std Dev:", np.std(ecg_signal))
        print("Contains NaN:", np.isnan(ecg_signal).any())
        print("Contains only zeros:", np.all(ecg_signal == 0))
        print("--------------------------------------\n")

        # If the entire signal is NaN, return a zero array
        if np.isnan(ecg_signal).all():
            print("⚠️ Warning: ECG signal is completely NaN. Returning zeros.")
            return np.zeros((500,))  # Return zeros if the entire signal is NaN

        # Replace NaN values with zero (or interpolate if needed)
        ecg_signal = np.nan_to_num(ecg_signal, nan=0.0)

        # Apply Neurokit2 cleaning only if the signal is valid
        try:
            clean_signal = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
            return clean_signal
        except Exception as e:
            print(f"⚠️ Warning: NeuroKit2 failed. Returning zeros instead. Error: {e}")
            return np.zeros((500,))  # Return a zero array if NeuroKit2 fails

    def quality(self, ecg_signal, sampling_rate):
        """Compute ECG quality using neurokit2's ecg_quality."""
        try:
            quality = nk.ecg_quality(ecg_signal, sampling_rate=sampling_rate, method="zhao2018")
            return quality
        except Exception as e:
            raise ValueError("Error computing ECG quality: " + str(e))

    def pqrst_peaks(self, ecg_signal, sampling_rate):
        """Find PQRST peaks using neurokit2's ecg_peaks and ecg_delineate."""
        accessor = 'ECG_' + self.peak + '_Peaks'
        try:
            _, waves_peak = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
            if self.peak != 'R':
                _, waves_peak = nk.ecg_delineate(ecg_signal, waves_peak, sampling_rate=sampling_rate, method="peak")
            return {accessor: waves_peak[accessor], 'sampling_rate': sampling_rate}
        except Exception as e:
            raise ValueError("Error finding PQRST peaks: " + str(e))

    def preprocess(self, data, sampling_rate, rpeaks=None, random_shift=False):
        """Preprocess the ECG data."""
        result = []
        qual = []

        ecg_clean = self.clean(data, sampling_rate)

        if rpeaks is None:
            rpeaks = self.pqrst_peaks(ecg_clean, sampling_rate)

        #temp = rpeaks['ECG_' + self.peak + '_Peaks']
        #temp = temp[(temp - self.onset) >= 0]
        #temp = temp[(temp + self.offset) < len(data)]

        rpeaks_raw = rpeaks['ECG_' + self.peak + '_Peaks']

        # Optional random shift
        if random_shift:
            rpeaks_raw = rpeaks_raw + int(np.random.normal(0, 256))

        # Filter out beats that would go out of bounds
        ind = (rpeaks_raw >= 0) & (rpeaks_raw < len(data))
        rpeaks_raw = rpeaks_raw[ind]

        for i in range(1, len(rpeaks_raw) - 1):
            k = rpeaks_raw[i]
            prev_rr = k - rpeaks_raw[i - 1]
            next_rr = rpeaks_raw[i + 1] - k

            total_window = int(0.5 * (prev_rr + next_rr))  # Or however much you want
            half_window = total_window // 2

            start = k - half_window
            end = k + half_window

            if start >= 0 and end < len(data):
                segment = data[start:end]

                # Resample to fixed length
                temp1 = nk.signal.signal_resample(
                    segment,
                    desired_length=self.final_length,
                    sampling_rate=sampling_rate,
                )

                result.append(temp1)
                qual.append("Excellent")

        # Convert to array
        result = np.array(result)

        if result.shape[0] == 0:
            print("⚠️ Warning: No valid samples extracted for scaling!")
            return np.zeros((1, self.final_length, 1)), ["[]"], np.array([False])


        # --- Min-Max Scale ---
        result = sp.minmax_scale(result, axis=1)

        # --- Reshape ---
        result = result.reshape(len(result), self.final_length, 1)

        return result, qual, ind
