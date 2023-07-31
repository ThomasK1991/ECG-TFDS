"""icentia11k dataset."""

import tensorflow_datasets as tfds
import pickle
import gzip
import os
from utils.preprocessing import Preprocess
import numpy as np

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
                }, length=500, doc='Single heartbeats of 1 second length'),
                'rhythm': tfds.features.ClassLabel(names=['N', 'AFIB', 'AFL']),
                'beat': tfds.features.ClassLabel(names=['N', 'PAC', 'PVC', 'Q']),
                'quality': tfds.features.ClassLabel(names=['Unacceptable', 'Barely acceptable', 'Excellent', '[]']),
            }),
            supervised_keys=('ecg', 'beat'),
            homepage='https://physionet.org/content/icentia11k-continuous-ecg/1.0/',
        )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract('https://physionet.org/static/published-projects/icentia11k-continuous-ecg/icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0.zip')

    # TODO(icentia11k): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0'),
    }

  def _generate_examples(self, path):

    preprocessor = Preprocess(250, 250, peak='R', final_length=500)

    sample_id = 1
    filename = os.path.join(path, ("%05d" % sample_id) + "_batched_lbls.pkl.gz")
    filename_ecg = os.path.join(path, ("%05d" % sample_id) + "_batched.pkl.gz")

    segments = pickle.load(gzip.open(filename))
    ecg = pickle.load(gzip.open(filename_ecg))

    for segment_id, segment_labels in enumerate(segments):

        beat_val = []
        beat_label = []
        rhythm_val = []
        rhythm_label = []

        for k in range(len(segment_labels['btype'])):
            temp = segment_labels['btype'][k]
            beat_val.append(temp)
            beat_label.append(np.full(len(temp), k))

        for k in range(len(segment_labels['rtype'])):
            temp = segment_labels['rtype'][k]
            rhythm_val.append(temp)
            rhythm_label.append(np.full(len(temp), k))

        beat_val = np.concatenate(beat_val) * 2
        beat_label = np.concatenate(beat_label)

        rhythm_val = np.concatenate(rhythm_val) * 2
        rhythm_label = np.concatenate(rhythm_label)

        ecg_clean, q = preprocessor.preprocess(data=ecg[segment_id], sampling_rate=250)

        for i, k in ecg_clean:
            yield 'key', {
                'ecg': {
                    'I': k,
                },
                'rhythm': 'N',
                'beat': 'N',
                'quality': q[i],
            }
