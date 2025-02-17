"""ptb dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import wfdb
import numpy as np
import ast
import sys
import os
from utils.preprocessing import Preprocess
from helper_code import *

# Ensure project path is added
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ptb dataset."""

    VERSION = tfds.core.Version('1.0.3')
    RELEASE_NOTES = {
        '1.0.3': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        self.classes = ['164889003', '164890007', '6374002', '426627000', '733534002',
                        '713427006', '270492004', '713426002', '39732003', '445118002',
                        '164947007', '251146004', '111975006', '698252002', '426783006',
                        '284470004', '10370003', '365413008', '427172004', '164917005',
                        '47665007', '427393009', '426177001', '427084000', '164934002',
                        '59931005']

        self.equivalent_classes = [['713427006', '59118001'],
                                   ['284470004', '63593006'],
                                   ['427172004', '17338001'],
                                   ['733534002', '164909002']]

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'ecg': tfds.features.Sequence({'I': np.float64}),
                'subject': tfds.features.Text(),
                'quality': tfds.features.ClassLabel(names=['Unacceptable', 'Barely acceptable', 'Excellent', '[]']),
                'nsamp': np.uint16,
                'leads': tfds.features.Sequence(tfds.features.Text()),
                'age': np.uint16,
                'gender': tfds.features.ClassLabel(names=['Male', 'Female']),
                'frequency': np.uint8,
                'diagnostic': tfds.features.Tensor(shape=(26,), dtype=np.bool_),
                'dx': tfds.features.Sequence(tfds.features.Text()),
            }),
            supervised_keys=None,
            homepage='https://physionet.org/content/ptb-xl/1.0.3/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = r"C:\Users\Thomas Kaprielian\Documents\4yp_Data\physionet.org\files\challenge-2021\1.0.3\training\ptb-xl\g1"

        if not os.path.exists(path):
            raise ValueError(f"The specified dataset path {path} does not exist.")

        header_files, recording_files = self.find_challenge_files(path)

        if len(header_files) == 0:
            raise ValueError(f"No .hea files found in {path}")

        return {
            'train': self._generate_examples(header_files, recording_files),
        }

    def find_challenge_files(self, data_directory):
        header_files = []
        recording_files = []
        for f in os.listdir(data_directory):
            root, extension = os.path.splitext(f)
            if not root.startswith('.') and extension == '.hea':
                header_file = os.path.join(data_directory, root + '.hea')
                recording_file = os.path.join(data_directory, root + '.mat')
                if os.path.isfile(header_file) and os.path.isfile(recording_file):
                    header_files.append(header_file)
                    recording_files.append(recording_file)
        return header_files, recording_files

    def aggregate_diagnostic(self, y_dic, path):
        tmp = []
        agg_df = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def get_nsamp(self, header):
        print("Header content:", header)
        return int(header.split('\n')[0].split(' ')[3])

    def replace_equivalent_classes(self,classes, equivalent_classes):
        for j, x in enumerate(classes):
            for multiple_classes in equivalent_classes:
                if x in multiple_classes:
                    classes[j] = multiple_classes[0]  # Use the first class as the representative class.
        return classes

    def _generate_examples(self, header_files, recording_files):
        """Yields examples from the local dataset."""
        for header_file, recording_file in zip(header_files, recording_files):
            hdr = load_header(header_file)

            tmp = {
                'header': header_file,
                'record': recording_file,
                'nsamp': self.get_nsamp(hdr),
                'leads': get_leads(hdr),
                'age': get_age(hdr),
                'sex': get_sex(hdr),
                'dx': get_labels(hdr),
                'fs': get_frequency(hdr),
                'target': np.zeros((26,))
            }

            # Convert gender to expected format
            if tmp['sex'] == 'M':
                tmp['sex'] = 'Male'
            elif tmp['sex'] == 'F':
                tmp['sex'] = 'Female'

            tmp['dx'] = self.replace_equivalent_classes(tmp['dx'],self.equivalent_classes)

            for dx in tmp['dx']:
                if dx in self.classes:
                    idx = self.classes.index(dx)
                    tmp['target'][idx] = 1

            # Load ECG signals (removing .mat extension)
            record = wfdb.rdsamp(recording_file[:-4])  # Remove '.mat'
            ecg_signal = record[0] if record else np.zeros((tmp['nsamp'], 1))  # Default to zeros if loading fails

            yield header_file, {
                'ecg': {'I': ecg_signal.flatten()},
                'subject': tmp['header'],
                'quality': 'Excellent',  # Placeholder value
                'nsamp': tmp['nsamp'],
                'leads': tmp['leads'],
                'age': tmp['age'],
                'gender': tmp['sex'],
                'diagnostic': tmp['target'],
                'dx': tmp['dx'],
            }
