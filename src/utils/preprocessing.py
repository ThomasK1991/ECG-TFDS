import sklearn.preprocessing as sp
import numpy as np
import neurokit2 as nk
from scipy import signal

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

        temp = rpeaks['ECG_' + self.peak + '_Peaks'] - self.onset
        if random_shift:
            temp = temp + int(np.random.normal(0, 256)) #.astype(int)
        ind = (temp >= 0) & (temp + self.window_length < len(data))
        temp = temp[ind]

        for k in temp:
            temp1 = nk.signal.signal_resample(
                data[k:(k + self.window_length)],
                desired_length=self.final_length,
                sampling_rate=sampling_rate,
            )
            result.append(temp1)
            qual.append('Excellent') #self.quality(temp1, sampling_rate))

        result = np.array(result)
        result = sp.minmax_scale(result, axis=1)
        result = result.reshape(len(result), self.final_length, 1)

        return result, qual, ind
