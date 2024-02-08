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

    VERSION = tfds.core.Version('1.0.3')
    RELEASE_NOTES = {
        '1.0.3': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'ecg': tfds.features.Sequence({
                    'I': np.float64,
                }, length=500, doc='Single heartbeats of 1 second length'),
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

        result = {}

        # TODO: Adjust the path to your needs - due to the size of the dataset (> 1 TB unzipped) no automatic
        #  download is provided
        path = '/mnt/sdb/icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0/'
        np.random.seed(42)
        ids = np.random.choice(range(11000), 100, replace=False)
        for k in ids:
            result.update({str(k): self._generate_examples(path, k)})

        return result

    def _generate_examples(self, path, subject):

        preprocessor = Preprocess(125, 125, peak='R', final_length=500)
        for segment in range(0, 50):
            try:
                t = subject // 1000
                filename = path + 'p' + f"{t :02d}" + '/p' + f"{subject :05d}" + '/p' + f"{subject :05d}" + '_s' + f"{segment :02d}"
                rec = wfdb.rdrecord(filename)
                ann = wfdb.rdann(filename, "atr")

                ann.symbol = np.array(ann.symbol)
                ann.sample = np.array(ann.sample)

                signal = rec.p_signal.flatten()
                rpeaks = pd.DataFrame(ann.sample[ann.symbol != '+'], columns=['ECG_R_Peaks'])
                labels = ann.symbol[ann.symbol != '+']

                ecg_clean, q, ind = preprocessor.preprocess(data=signal, sampling_rate=ann.fs, rpeaks=rpeaks)
                labels = labels[ind]

                for i, k in enumerate(ecg_clean[:, :, 0]):
                    yield str(subject) + '_' + str(segment) + '_' + str(i), {
                        'ecg': {
                            'I': k,
                        },
                        'patient': int(subject),
                        'segment': int(segment),
                        # TODO: the rhythm information is now statically set to N
                        'rhythm': 'N',
                        'beat': labels[i],
                        'quality': q[i],
                    }
            except:
                continue
