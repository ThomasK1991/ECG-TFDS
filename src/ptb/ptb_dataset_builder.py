"""ptb dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import wfdb
import numpy as np
import ast
from utils.preprocessing import Preprocess
import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for ptb dataset."""

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
                }),
                'quality': tfds.features.ClassLabel(names=['Unacceptable', 'Barely acceptable', 'Excellent', '[]']),
                'age': np.uint8,
                'gender': np.uint8,
                'diagnostic': tfds.features.ClassLabel(names=['STTC', 'NORM', 'MI', 'HYP', 'CD', 'NAV']),
            }),
            supervised_keys=None,
            homepage='https://physionet.org/content/ptb-xl/1.0.3/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(
            'https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip')
        return {
            'train': self._generate_examples(
                path / 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'),
        }

    def aggregate_diagnostic(self, y_dic, path):
        tmp = []
        agg_df = pd.read_csv(path + '/' + 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def _generate_examples(self, path):
        """Yields examples."""
        preprocessor = Preprocess(250, 250, peak='R', final_length=500)

        metadata = pd.read_csv(str(path) + '/' + 'ptbxl_database.csv', index_col='ecg_id')
        metadata.scp_codes = metadata.scp_codes.apply(lambda x: ast.literal_eval(x))
        metadata['diagnostic_superclass'] = metadata.scp_codes.apply(self.aggregate_diagnostic, path=str(path))

        for index, row in metadata.iterrows():
            data = wfdb.rdsamp(str(path) + '/' + row['filename_hr'])[0][:, 0]
            data_prep, q, ind = preprocessor.preprocess(data=data, sampling_rate=500)

            #for j, k in enumerate(data_prep):
            key = str(row['patient_id']) + "_" + str(index)
            diagnostic = "NAV" if len(row['diagnostic_superclass']) == 0 else row['diagnostic_superclass'][0]
            yield key, {
                'ecg': {
                    'I':np.median(data_prep, axis=0).flatten(),
                },
                'quality': str(q[0]),
                'age': row['age'],
                'gender': row['sex'],
                'diagnostic': diagnostic,
            }
