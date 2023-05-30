"""zheng dataset."""

import tensorflow_datasets as tfds
from utils.preprocessing import Preprocess
import numpy as np
import pandas as pd
import urllib.request
import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for zheng dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'ecg': tfds.features.Sequence({
                    'I': np.float64,
                }, length=500, doc='Single heartbeats of 1 second length'),
                'rhythm': tfds.features.ClassLabel(
                    names_file='./metadata/rhythm.txt'
                ),
                'beat': tfds.features.ClassLabel(
                    names_file='./metadata/beat.txt'
                ),
                'quality': tfds.features.ClassLabel(names=['Unacceptable', 'Barely acceptable', 'Excellent', '[]']),
                'age': np.uint8,
                'gender': tfds.features.ClassLabel(names=['MALE', 'FEMALE']),
                'ventricular_rate': np.uint8,
                'atrial_rate': np.uint8,
                'qrs_duration': np.uint8,
                'qt_interval': np.uint8,
                'qt_corrected': np.uint8,
                'r_axis': np.uint8,
                't_axis': np.uint8,
                'qrs_count': np.uint8,
                'q_onset': np.uint8,
                'q_offset': np.uint8,
                't_offset': np.uint8,
            }),
            supervised_keys=('ecg', 'rhythm'),
            homepage='https://figshare.com/collections/ChapmanECG/4560497/2',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract('https://figshare.com/ndownloader/files/15652862')
        urllib.request.urlretrieve("https://figshare.com/ndownloader/files/15653771", "./Diagnostics.xlsx")
        return {
            'train': self._generate_examples(path / 'ECGDataDenoised'),
        }

    def _generate_examples(self, path):

        preprocessor = Preprocess(250, 250, peak='R', final_length=500)
        metadata = pd.read_excel('./Diagnostics.xlsx')
        for index, row in metadata.iterrows():
            try:
                f = str(path) + '/' + row['FileName'] + '.csv'
                data = pd.read_csv(f, delimiter=',', header=None, usecols=[0], names=['I'])
                data = data['I'].to_numpy().flatten()
                data_prep, q = preprocessor.preprocess(data=data, sampling_rate=500)
                for j, k in enumerate(data_prep):
                    key = row['FileName'] + "_" + str(j)
                    yield key, {
                        'ecg': {
                            'I': k.flatten(),
                        },
                        'rhythm': row['Rhythm'],
                        'beat': row['Beat'],
                        'quality': str(q[j]),
                        'age': row['PatientAge'],
                        'gender': row['Gender'],
                        'ventricular_rate': row['VentricularRate'],
                        'atrial_rate': row['AtrialRate'],
                        'qrs_duration': row['QRSDuration'],
                        'qt_interval': row['QTInterval'],
                        'qt_corrected': row['QTCorrected'],
                        'r_axis': row['RAxis'],
                        't_axis': row['TAxis'],
                        'qrs_count': row['QRSCount'],
                        'q_onset': row['QOnset'],
                        'q_offset': row['QOffset'],
                        't_offset': row['TOffset'],
                    }
            except:
                pass
