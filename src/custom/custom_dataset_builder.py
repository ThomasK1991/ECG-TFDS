"""custom dataset."""

import tensorflow_datasets as tfds
from glob import glob
import pandas as pd
import numpy as np
from utils.preprocessing import Preprocess

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for custom dataset."""

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
                'quality': tfds.features.ClassLabel(names=['Unacceptable', 'Barely acceptable', 'Excellent', '[]']),
            }),
            supervised_keys=('ecg', 'quality'),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO: Adjust to where your custom data set is stored
        path_data = './data/'
        return {
            'train': self._generate_examples(path_data),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO: Adjust to your own needs
        onset = 65
        offset = 65
        preprocessor = Preprocess(onset, offset, peak='R', final_length=500)
        for name in glob(path + '**/*.csv', recursive=True):
            try:
                # NOTE (HELP): The implementation expects the first column to be the ECG channel of interest.
                df = pd.read_csv(name)
                # TODO: Adjust the sampling rate
                data_prep, q, ind = preprocessor.preprocess(data=df['ecg'], sampling_rate=130)
                for j, k in enumerate(data_prep):
                    key = name + "_" + str(j)
                    yield key, {
                        'ecg': {
                            'I': k.flatten(),
                        },
                        'quality': str(q[j]),
                    }
            except Exception as e:
                print(e)
                pass
