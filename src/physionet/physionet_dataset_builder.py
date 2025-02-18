"""physionet dataset."""

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
                'ecg': tfds.features.Sequence({
                    'I': np.float64, 'II': np.float64, 'III': np.float64,
                    'aVR': np.float64, 'aVL': np.float64, 'aVF': np.float64,
                    'V1': np.float64, 'V2': np.float64, 'V3': np.float64,
                    'V4': np.float64, 'V5': np.float64, 'V6': np.float64,
                }),
                'subject': tfds.features.Text(),
                'quality': tfds.features.ClassLabel(names=['Unacceptable', 'Barely acceptable', 'Excellent', '[]']),
                'nsamp': np.uint16,
                'lead': tfds.features.Text(),
                'age': np.uint16,
                'gender': tfds.features.ClassLabel(names=['Male', 'Female', 'Unknown']),
                'frequency': np.uint16,
                'diagnostic': tfds.features.Tensor(shape=(26,), dtype=np.float32),
                'dx': tfds.features.Sequence(tfds.features.Text()),
            }),
            supervised_keys=None,
            homepage='https://physionet.org/content/ptb-xl/1.0.3/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = r"C:\Users\Thomas Kaprielian\Documents\4yp_Data\test_VAE"

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
    
    def expand_leads(self,recording,input_leads):
        output = np.zeros((12, recording.shape[1]))
        twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
        twelve_leads = [k.lower() for k in twelve_leads]

        input_leads = [k.lower() for k in input_leads]
        output_leads = np.zeros((12,))
        for i,k in enumerate(input_leads):
            idx = twelve_leads.index(k)
            output[idx,:] = recording[i,:]
            output_leads[idx] = 1
        return output,output_leads

    def _generate_examples(self, header_files, recording_files):
        """Yields segmented ECG examples with preprocessing applied."""

        # Initialize Preprocessor (Chopping ECG into peaks)
        preprocessor = Preprocess(250, 250, peak='R', final_length=500)

        for header_file, recording_file in zip(header_files, recording_files):
            # Load metadata from header file
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

            # Convert age: Ensure it's an integer and replace NaN with 0
            age = int(tmp['age']) if not np.isnan(tmp['age']) else 0

            # Convert gender: 'M' -> 'Male', 'F' -> 'Female'
            gender = 'Male' if tmp['sex'] == 'M' else 'Female' if tmp['sex'] == 'F' else 'Unknown'

            # Replace equivalent diagnostic classes
            tmp['dx'] = self.replace_equivalent_classes(tmp['dx'], self.equivalent_classes)

            # Convert diagnostic labels to one-hot encoding
            for dx in tmp['dx']:
                if dx in self.classes:
                    idx = self.classes.index(dx)
                    tmp['target'][idx] = 1

            # Load full ECG signal
            record_path = os.path.splitext(recording_file)[0]  # Remove '.mat' extension
            try:
                record = wfdb.rdsamp(record_path)
                ecg_data = record[0].T  # Transpose (num_leads, num_samples)
                available_leads = tmp['leads']
            except Exception as e:
                print(f"Error reading {record_path}: {e}")
                ecg_data = np.zeros((12, tmp['nsamp']))  # Default to zeros if loading fails
                available_leads = []



            # Apply Preprocessing (Segment ECG into peaks for each lead)
            segmented_ecgs = []
            qualities = []
            for lead_signal in ecg_data:
                data_prep, q, _ = preprocessor.preprocess(data=lead_signal, sampling_rate=tmp['fs'])
                segmented_ecgs.append(data_prep)
                qualities.append(q)

            # Ensure segmentation length consistency across leads
            num_segments = min(len(seg) for seg in segmented_ecgs)
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

            # Assign a unique key and yield each segmented ECG as a separate entry
            for j in range(num_segments):
                key = f"{os.path.basename(header_file)}_{j}"  # Unique key per segment

                    # Extract only the processed segments (without manual zero padding)
                unexpanded_ecg = np.array([
                    seg[j].flatten() for seg in segmented_ecgs if seg is not None
                ])

                # Expand leads at the very end
                expanded_ecg, output_leads = self.expand_leads(unexpanded_ecg, available_leads)

                # Store ECG as a dictionary with 12 leads
                ecg_dict = {lead_names[i]: expanded_ecg[i].flatten() for i in range(12)}

                yield key, {
                    'ecg': ecg_dict,  # Each segment contains a full 12-lead ECG
                    'subject': os.path.basename(header_file),  # Use filename as subject ID
                    'quality': str(qualities[0][j]),  # Assign quality from the first lead
                    'age': age,  # Ensure integer age
                    'gender': gender,  # Convert gender format
                    'diagnostic': tmp['target'].astype(np.float32),  # Ensure float32 for TFDS
                    'dx': tmp['dx'],
                    'nsamp': len(segmented_ecgs[0][j]),  # Number of samples per segment
                    'lead': '12-lead',  # Indicate full 12-lead ECG
                    'frequency': tmp['fs']
                }