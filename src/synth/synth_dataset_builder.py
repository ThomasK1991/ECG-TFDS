"""synth dataset."""

import tensorflow_datasets as tfds
from utils.preprocessing import Preprocess
import numpy as np
import itertools
import subprocess

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for synth dataset."""

    VERSION = tfds.core.Version('1.0.3')
    RELEASE_NOTES = {
        '1.0.3': 't + p wave',
    }

    def runcmd(self, cmd, verbose=False, *args, **kwargs):
        # https://www.scrapingbee.com/blog/python-wget/
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        std_out, std_err = process.communicate()
        if verbose:
            print(std_out.strip(), std_err)
        pass

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'ecg': tfds.features.Sequence({
                    'I': np.float64,
                }, length=500, doc='Single heartbeats of 1 second length'),
                'quality': tfds.features.ClassLabel(names=['Unacceptable', 'Barely acceptable', 'Excellent', '[]']),
                't_height': np.float64,
                'p_height': np.float64,
            }),
            supervised_keys=('ecg', 'quality'),  # Set to `None` to disable
            homepage='https://physionet.org/content/ecgsyn/1.0.0/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):

        self.runcmd('wget -r -N -c -np https://physionet.org/files/ecgsyn/1.0.0/')
        # TODO: uncomment if you like to use the arrhythmia ecg generator for dataset creation
        # self.runcmd('wget -r -N -c -np https://physionet.org/files/ecg-ppg-simulator-arrhythmia/1.3.1/')

        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self):
        import matlab.engine

        preprocessor = Preprocess(250, 250, peak='R', final_length=500)
        eng = matlab.engine.start_matlab()
        path = eng.genpath('./physionet.org/files/ecgsyn/1.0.0/Matlab/')
        eng.addpath(path, nargout=0)

        for t_extrema in np.linspace(-2, 2,10):
            for p_extrema in np.linspace(-2, 2, 10):
                res = eng.ecgsyn(
                    matlab.double(512),  # sfecg: ECG sampling frequency [256 Hertz]
                    matlab.double(50),  # N: approximate number of heart beats [256]
                    matlab.double(0.01),  # Anoise: Additive uniformly distributed measurement noise [0 mV]
                    matlab.double(60),  # hrmean: Mean heart rate [60 beats per minute]
                    matlab.double(2),  # hrstd: Standard deviation of heart rate [1 beat per minute]
                    matlab.double(0.5),  # lfhfratio: LF/HF ratio [0.5]
                    matlab.double(512),  # sfint: Internal sampling frequency [256 Hertz]
                    # Order of extrema: [P Q R S T]
                    matlab.double([-70, -15, 0, 15, 100]),  # ti = angles of extrema [-70 -15 0 15 100] degrees
                    matlab.double([p_extrema, -5, 30, -7.5, t_extrema]),  # ai = z-position of extrema [1.2 -5 30 -7.5 0.75]
                    matlab.double([0.25, 0.1, 0.1, 0.1, 0.4]),  # bi = Gaussian width of peaks [0.25 0.1 0.1 0.1 0.4]
                )
                data_prep, q, ind = preprocessor.preprocess(data=np.array(res)[:,0], sampling_rate=512)
                for j, k in enumerate(data_prep):
                    key = str(hash('key_' + str(j) + '_' + str(t_extrema) + '_' + str(p_extrema)))
                    yield key, {
                        'ecg': {
                            'I': k.flatten(),
                        },
                        'quality': q[0],
                        't_height': np.float64(t_extrema),
                        'p_height': np.float64(p_extrema),
                    }
