"""icentia11k dataset."""

import tensorflow_datasets as tfds
import wfdb
from utils.preprocessing import Preprocess
import numpy as np
import sys
import os
import pandas as pd

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for icentia11k dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(icentia11k): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'ecg': tfds.features.Sequence({
                    'I': np.float64,
                }, length=100, doc='Single heartbeats of 1 second length'),
                'patient': np.int64,
                'segment': np.int64,
                'rhythm': tfds.features.ClassLabel(names=['N', 'AFIB', 'AFL']),
                'beat': tfds.features.ClassLabel(names=['N', 'Q', 'S', 'V']),
                'quality': tfds.features.ClassLabel(names=['Unacceptable', 'Barely acceptable', 'Excellent', '[]']),
            }),
            supervised_keys=('ecg', 'beat'),
            homepage='https://physionet.org/content/icentia11k-continuous-ecg/1.0/',
        )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # path = dl_manager.download_and_extract('https://physionet.org/static/published-projects/icentia11k-continuous-ecg/icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0.zip')
    path = './data/'

    segment_list = [
        ['p00045', 's00'],
        ['p00045', 's01'],
        ['p00002', 's00'],
        ['p00002', 's01'],
        ['p00002', 's02'],
        ['p00002', 's04'],
    ]
    return {
        'train': self._generate_examples(path, segment_list),
    }

  def _generate_examples(self, path, segments):

    preprocessor = Preprocess(125, 125, peak='R', final_length=100)

    for k in segments:
        patient, segment = k
        filename = path + str(patient) + '_' + str(segment)
        rec = wfdb.rdrecord(filename)
        ann = wfdb.rdann(filename, "atr")

        ann.symbol = np.array(ann.symbol)
        ann.sample = np.array(ann.sample)

        signal = rec.p_signal.flatten()
        rpeaks = pd.DataFrame(ann.sample[ann.symbol != '+'], columns=['ECG_R_Peaks'])
        labels = ann.symbol[ann.symbol != '+']

        ecg_clean, q, ind = preprocessor.preprocess(data=signal, sampling_rate=ann.fs, rpeaks=rpeaks)
        labels = labels[ind]

        for i, k in enumerate(ecg_clean[:,:,0]):
            yield str(patient) + '_' + str(segment) + '_' + str(i), {
                'ecg': {
                    'I': k,
                },
                'patient': int(patient[1:]),
                'segment': int(segment[1:]),
                'rhythm': 'N',
                'beat': labels[i],
                'quality': q[i],
            }
