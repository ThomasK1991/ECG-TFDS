"""ptb dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import wfdb
import numpy as np
import ast
from src.utils.preprocessing import Preprocess


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

    def aggregate_diagnostic(self, path, y_dic):
        tmp = []
        agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    def _generate_examples(self, path):
        """Yields examples."""
        preprocessor = Preprocess(250, 250, peak='R', final_length=500)

        # TODO(ptb): Yields (key, example) tuples from the dataset
        metadata = pd.read_csv(str(path) + '/' + 'ptbxl_database.csv', index_col='ecg_id')
        metadata.scp_codes = metadata.scp_codes.apply(lambda x: ast.literal_eval(x))
        # metadata['diagnostic_superclass'] = metadata.scp_codes.apply(self.aggregate_diagnostic)

        for index, row in metadata.iterrows():
            data = wfdb.rdsamp(str(path) + '/' + row['filename_hr'])[0][:, 0]
            data_prep, q = preprocessor.preprocess(data=data, sampling_rate=500)

            for j, k in enumerate(data_prep):
                key = str(row['patient_id']) + "_" + str(index) + "_" + str(j)
                yield key, {
                    'ecg': {
                        'I': k.flatten(),
                    },
                    'quality': str(q[j]),
                    'age': row['age'],
                }
