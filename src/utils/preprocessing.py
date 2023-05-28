import sklearn.preprocessing as sp
import numpy as np
import neurokit2 as nk


class Preprocess:
    def __init__(self, onset: int, offset: int, final_length: int = None, peak='R'):
        self.onset = onset
        self.offset = offset
        self.window_length = self.onset + self.offset
        self.final_length = final_length or self.window_length
        self.peak = peak

    def clean(self, ecg_signal, sampling_rate):
        try:
            clean = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
            return clean
        except:
            return []

    def quality(self, ecg_signal, sampling_rate):
        try:
            quality = nk.ecg_quality(ecg_signal, sampling_rate=sampling_rate, method="zhao2018")
            return quality
        except:
            return []

    def pqrst_peaks(self, ecg_signal, sampling_rate):
        accessor = 'ECG_' + self.peak + '_Peaks'
        try:
            _, waves_peak = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
            if self.peak != 'R':
                _, waves_peak = nk.ecg_delineate(ecg_signal, waves_peak, sampling_rate=sampling_rate, method="peak")
            return {accessor: waves_peak[accessor], 'sampling_rate': 500}
        except Exception as e:
            return {accessor: np.empty(0), 'sampling_rate': 500}

    def preprocess(self, data, sampling_rate):
        result = []
        qual = []

        ecg_clean = self.clean(data, sampling_rate)
        rpeaks = self.pqrst_peaks(ecg_clean, sampling_rate)

        temp = rpeaks['ECG_' + self.peak + '_Peaks'] - self.onset
        temp = temp[(temp >= 0) & (temp + self.window_length < len(data))]

        for k in temp:
            temp1 = nk.signal.signal_resample(data[k:(k + self.window_length)], desired_length=self.final_length, sampling_rate=sampling_rate)
            result.append(temp1)
            qual.append(self.quality(temp1, sampling_rate))

        result = np.array(result)
        result = sp.minmax_scale(result, axis=1)
        result = result.reshape(len(result), self.final_length, 1)

        return result, qual
