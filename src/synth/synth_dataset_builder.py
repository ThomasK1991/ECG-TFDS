"""synth dataset."""

import tensorflow_datasets as tfds
from utils.preprocessing import Preprocess
import numpy as np
import subprocess
from itertools import product
import random
import string
import copy


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for synth dataset."""

    VERSION = tfds.core.Version('1.0.11')
    RELEASE_NOTES = {
        '1.0.11': 'Change all parameters.',
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

    def key_generator(size=128, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'ecg': tfds.features.Sequence({
                    'I': np.float64,
                }, length=500, doc='Single heartbeats of 1 second length'),
                'quality': tfds.features.ClassLabel(names=['Unacceptable', 'Barely acceptable', 'Excellent', '[]']),
                'p_height': np.float64,
                'q_height': np.float64,
                'r_height': np.float64,
                's_height': np.float64,
                't_height': np.float64,
                'p_angle': np.float64,
                'q_angle': np.float64,
                'r_angle': np.float64,
                's_angle': np.float64,
                't_angle': np.float64,
                'p_width': np.float64,
                'q_width': np.float64,
                'r_width': np.float64,
                's_width': np.float64,
                't_width': np.float64,
                's_fecg': np.float64,
                'N': np.float64,
                'anoise': np.float64,
                'hrmean': np.float64,
                'hrstd': np.float64,
                'lfhfratio': np.float64,
                'sfint': np.float64,
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
        parameters = {
            'p_height': 1.2,
            'q_height': -5.0,
            'r_height': 30.0,
            's_height': -7.5,
            't_height': 0.75,
            'p_angle': -70.0,
            'q_angle': -15.0,
            'r_angle': 0.0,
            's_angle': 15.0,
            't_angle': 100.0,
            'p_width': 0.25,
            'q_width': 0.1,
            'r_width': 0.1,
            's_width': 0.1,
            't_width': 0.4,
            'sfecg': 500,
            'N': 50,
            'anoise': 0.02,
            'hrmean': 60,
            'hrstd': 7,
            'lfhfratio': 0.5,
            'sfint': 500,
        }
        # TODO: In case of all vs. all generation (adapt the values to be arrays, e.g., t_width = np.linspace(0.2,0.6, 10),)
        # combinations = [dict(zip(parameters, values)) for values in product(*parameters.values())]

        toggle = {
            'p_height': np.linspace(-2,2, 10),
            'q_height': np.linspace(-10,0, 10),
            'r_height': np.linspace(5,50, 10),
            's_height': np.linspace(-15,15, 10),
            't_height': np.linspace(-2,2, 10),
            'p_angle': np.linspace(-120,-20, 10),
            'q_angle': np.linspace(-25,-15, 10),
            'r_angle': np.linspace(-5,5, 10),
            's_angle': np.linspace(5,25, 10),
            't_angle': np.linspace(75,125, 10),
            'p_width': np.linspace(0.1,4, 10),
            'q_width': np.linspace(0.01,0.3, 10),
            'r_width': np.linspace(0.01,0.3, 10),
            's_width': np.linspace(0.01,0.3, 10),
            't_width': np.linspace(0.2,0.6, 10),
        }


        combinations = []
        for k, val in toggle.items():
            for j in val:
                temp = copy.deepcopy(parameters)
                temp[k] = j
                combinations.append(temp)

        for c in combinations:
            try:
                res = eng.ecgsyn(
                    matlab.double(c['sfecg']),
                    matlab.double(c['N']),
                    matlab.double(c['anoise']),
                    matlab.double(c['hrmean']),
                    matlab.double(c['hrstd']),
                    matlab.double(c['lfhfratio']),
                    matlab.double(c['sfint']),
                    matlab.double([c['p_angle'], c['q_angle'], c['r_angle'], c['s_angle'], c['t_angle']]),
                    matlab.double([c['p_height'], c['q_height'], c['r_height'], c['s_height'], c['t_height']]),
                    matlab.double([c['p_width'], c['q_width'], c['r_width'], c['s_width'], c['t_width']]),
                )
                data_prep, q, ind = preprocessor.preprocess(data=np.array(res)[:, 0], sampling_rate=500)
                for j, k in enumerate(data_prep):
                    key = str(random.getrandbits(128))
                    yield key, {
                        'ecg': {
                            'I': k.flatten(),
                        },
                        'quality': q[0],
                        'p_height': np.float64(c['p_height']),
                        'q_height': np.float64(c['q_height']),
                        'r_height': np.float64(c['r_height']),
                        's_height': np.float64(c['s_height']),
                        't_height': np.float64(c['t_height']),
                        'p_angle': np.float64(c['p_angle']),
                        'q_angle': np.float64(c['q_angle']),
                        'r_angle': np.float64(c['r_angle']),
                        's_angle': np.float64(c['s_angle']),
                        't_angle': np.float64(c['t_angle']),
                        'p_width': np.float64(c['p_width']),
                        'q_width': np.float64(c['q_width']),
                        'r_width': np.float64(c['r_width']),
                        's_width': np.float64(c['s_width']),
                        't_width': np.float64(c['t_width']),
                        's_fecg': np.float64(c['sfecg']),
                        'N': np.float64(c['N']),
                        'anoise': np.float64(c['anoise']),
                        'hrmean': np.float64(c['hrmean']),
                        'hrstd': np.float64(c['hrstd']),
                        'lfhfratio': np.float64(c['lfhfratio']),
                        'sfint': np.float64(c['sfint']),
                    }
            except Exception as e:
                print(e)
                continue
