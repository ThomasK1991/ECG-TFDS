"""medalcare dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import glob
from utils.preprocessing import Preprocess


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for medalcare dataset."""

    VERSION = tfds.core.Version('1.0.5')
    RELEASE_NOTES = {
        '1.0.5': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'ecg': tfds.features.Sequence({
                    'I': np.float64,
                }),
                'subject': np.int32,
                'diagnosis': tfds.features.ClassLabel(
                    names=['avblock', 'fam', 'iab', 'lae', 'lbbb', 'mi', 'rbbb', 'sinus']),
                'subdiagnosis': tfds.features.ClassLabel(
                    names=['None', 'examples', 'LAD_0.3', 'LAD_1.0', 'LCX_0.3_ant', 'LCX_0.3_post', 'LCX_1.0_ant', 'LCX_1.0_post',
                           'RCA_0.3', 'RCA_1.0']),
                'v_G.lungs': np.float64,
                'v_G.ischemia': np.float64,
                'v_G.torso': np.float64,
                'v_G.atria': np.float64,
                'v_G.et': np.float64,
                'v_G.el': np.float64,
                'v_G.en': np.float64,
                'v_G.it': np.float64,
                'v_G.in': np.float64,
                'v_G.blood': np.float64,
                'v_G.il': np.float64,
                'v_rvendo.z_min': np.float64,
                'v_rvendo.z_max': np.float64,
                'v_lvendo.z_max': np.float64,
                'v_lvendo.z_min': np.float64,
                'v_fibers.alpha_epi': np.float64,
                'v_fibers.alpha_endo': np.float64,
                'v_APD.min': np.float64,
                'v_APD.v_d': np.float64,
                'v_APD.z_d': np.float64,
                'v_APD.antpost_d': np.float64,
                'v_APD.max': np.float64,
                'v_APD.rho_d': np.float64,
                'v_cv.rvmyo_s_r': np.float64,
                'v_cv.lvmyo_s_r': np.float64,
                'v_cv.lvmyo_n_r': np.float64,
                'v_cv.lvendo_s_r': np.float64,
                'v_cv.rvmyo_n_r': np.float64,
                'v_cv.rvendo_n_r': np.float64,
                'v_cv.lvmyo_f': np.float64,
                'v_cv.lvendo_n_r': np.float64,
                'v_cv.lvendo_f': np.float64,
                'v_cv.rvendo_s_r': np.float64,
                'v_cv.rvmyo_f': np.float64,
                'v_cv.rvendo_f': np.float64,
                'v_stim[0].z': np.float64,
                'v_stim[0].thr': np.float64,
                'v_stim[0].phi': np.float64,
                'v_stim[0].ven': np.float64,
                'v_stim[0].rho_eps': np.float64,
                'v_stim[0].time': np.float64,
                'v_stim[0].rho': np.float64,
                'v_stim[1].phi': np.float64,
                'v_stim[1].rho': np.float64,
                'v_stim[1].time': np.float64,
                'v_stim[1].rho_eps': np.float64,
                'v_stim[1].ven': np.float64,
                'v_stim[1].thr': np.float64,
                'v_stim[1].z': np.float64,
                'v_stim[2].rho': np.float64,
                'v_stim[2].z': np.float64,
                'v_stim[2].ven': np.float64,
                'v_stim[2].thr': np.float64,
                'v_stim[2].phi': np.float64,
                'v_stim[2].rho_eps': np.float64,
                'v_stim[2].time': np.float64,
                'v_stim[3].time': np.float64,
                'v_stim[3].rho': np.float64,
                'v_stim[3].z': np.float64,
                'v_stim[3].rho_eps': np.float64,
                'v_stim[3].phi': np.float64,
                'v_stim[3].ven': np.float64,
                'v_stim[3].thr': np.float64,
                'v_stim[4].ven': np.float64,
                'v_stim[4].rho': np.float64,
                'v_stim[4].rho_eps': np.float64,
                'v_stim[4].time': np.float64,
                'v_stim[4].z': np.float64,
                'v_stim[4].phi': np.float64,
                'v_stim[4].thr': np.float64,
                'v_activate': tfds.features.ClassLabel(names=['True', 'False']),
                'v_G.fibrosis': np.float64,
                'v_RA.z': np.float64,
                'v_RA.rho': np.float64,
                'v_RA.phi': np.float64,
                'v_LA.z': np.float64,
                'v_LA.rho': np.float64,
                'v_LA.phi': np.float64,
                'v_RL.z': np.float64,
                'v_RL.rho': np.float64,
                'v_RL.phi': np.float64,
                'v_LL.z': np.float64,
                'v_LL.rho': np.float64,
                'v_LL.phi': np.float64,
                'v_V1.z': np.float64,
                'v_V1.rho': np.float64,
                'v_V1.phi': np.float64,
                'v_V2.z': np.float64,
                'v_V2.rho': np.float64,
                'v_V2.phi': np.float64,
                'v_V3.z': np.float64,
                'v_V3.rho': np.float64,
                'v_V3.phi': np.float64,
                'v_V4.z': np.float64,
                'v_V4.rho': np.float64,
                'v_V4.phi': np.float64,
                'v_V5.z': np.float64,
                'v_V5.rho': np.float64,
                'v_V5.phi': np.float64,
                'v_V6.z': np.float64,
                'v_V6.rho': np.float64,
                'v_V6.phi': np.float64,
                'v_purkinje': tfds.features.ClassLabel(names=['True', 'False']),
                'v_suffix': np.float64,
                # 'geo.atria': string,
                # 'geo.torso': string,
                'a_G.torso': np.float64,
                'a_rot.x': np.float64,
                'a_rot.y': np.float64,
                'a_rot.z': np.float64,
                'a_transl.x': np.float64,
                'a_transl.y': np.float64,
                'a_transl.z': np.float64,
                'a_cv_t.BulkTissue': np.float64,
                'a_cv_t.CristaTerminalis': np.float64,
                'a_cv_t.PectinateMuscles': np.float64,
                'a_cv_t.BachmannsBundle': np.float64,
                'a_cv_t.InferiorIsthmus': np.float64,
                'a_ar.BulkTissue': np.float64,
                'a_ar.CristaTerminalis': np.float64,
                'a_ar.PectinateMuscles': np.float64,
                'a_ar.BachmannsBundle': np.float64,
                'a_ar.InferiorIsthmus': np.float64,
                'a_fibperc': np.float64,
            }),
            supervised_keys=None,  # Set to `None` to disable
            homepage='https://www.nature.com/articles/s41597-023-02416-4',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        path = dl_manager.download_and_extract('https://zenodo.org/record/8068944/files/MedalCare-XL.zip?download=1')

        return {
            'train': self._generate_examples(path / 'MedalCare-XL', 'train'),
            'validation': self._generate_examples(path / 'MedalCare-XL', 'validation'),
            'test': self._generate_examples(path / 'MedalCare-XL', 'test'),
        }

    def _generate_examples(self, path, mode):
        """Yields examples."""

        allowed = ['v_G.lungs', 'v_G.ischemia', 'v_G.torso', 'v_G.atria', 'v_G.et', 'v_G.el', 'v_G.en', 'v_G.it',
                   'v_G.in', 'v_G.blood', 'v_G.il', 'v_rvendo.z_min', 'v_rvendo.z_max', 'v_lvendo.z_max',
                   'v_lvendo.z_min', 'v_fibers.alpha_epi', 'v_fibers.alpha_endo', 'v_APD.min', 'v_APD.v_d', 'v_APD.z_d',
                   'v_APD.antpost_d', 'v_APD.max', 'v_APD.rho_d', 'v_cv.rvmyo_s_r', 'v_cv.lvmyo_s_r', 'v_cv.lvmyo_n_r',
                   'v_cv.lvendo_s_r', 'v_cv.rvmyo_n_r', 'v_cv.rvendo_n_r', 'v_cv.lvmyo_f', 'v_cv.lvendo_n_r',
                   'v_cv.lvendo_f', 'v_cv.rvendo_s_r', 'v_cv.rvmyo_f', 'v_cv.rvendo_f', 'v_stim[0].z', 'v_stim[0].thr',
                   'v_stim[0].phi', 'v_stim[0].ven', 'v_stim[0].rho_eps', 'v_stim[0].time', 'v_stim[0].rho',
                   'v_stim[1].phi', 'v_stim[1].rho', 'v_stim[1].time', 'v_stim[1].rho_eps', 'v_stim[1].ven',
                   'v_stim[1].thr', 'v_stim[1].z', 'v_stim[2].rho', 'v_stim[2].z', 'v_stim[2].ven', 'v_stim[2].thr',
                   'v_stim[2].phi', 'v_stim[2].rho_eps', 'v_stim[2].time', 'v_stim[3].time', 'v_stim[3].rho',
                   'v_stim[3].z', 'v_stim[3].rho_eps', 'v_stim[3].phi', 'v_stim[3].ven', 'v_stim[3].thr',
                   'v_stim[4].ven', 'v_stim[4].rho', 'v_stim[4].rho_eps', 'v_stim[4].time', 'v_stim[4].z',
                   'v_stim[4].phi', 'v_stim[4].thr', 'v_activate', 'v_G.fibrosis', 'v_RA.z', 'v_RA.rho', 'v_RA.phi',
                   'v_LA.z', 'v_LA.rho', 'v_LA.phi', 'v_RL.z', 'v_RL.rho', 'v_RL.phi', 'v_LL.z', 'v_LL.rho', 'v_LL.phi',
                   'v_V1.z', 'v_V1.rho', 'v_V1.phi', 'v_V2.z', 'v_V2.rho', 'v_V2.phi', 'v_V3.z', 'v_V3.rho', 'v_V3.phi',
                   'v_V4.z', 'v_V4.rho', 'v_V4.phi', 'v_V5.z', 'v_V5.rho', 'v_V5.phi', 'v_V6.z', 'v_V6.rho', 'v_V6.phi',
                   'v_suffix', 'v_purkinje', 'a_G.torso', 'a_rot.x', 'a_rot.y', 'a_rot.z', 'a_transl.x', 'a_transl.y',
                   'a_transl.z', 'a_cv_t.BulkTissue', 'a_cv_t.CristaTerminalis', 'a_cv_t.PectinateMuscles',
                   'a_cv_t.BachmannsBundle', 'a_cv_t.InferiorIsthmus', 'a_ar.BulkTissue', 'a_ar.CristaTerminalis',
                   'a_ar.PectinateMuscles', 'a_ar.BachmannsBundle', 'a_ar.InferiorIsthmus', 'a_fibperc']

        path = str(path) + '/WP2_largeDataset_ParameterFiles/**/' + mode + '/**/'
        files = glob.glob(path + '/*.txt', recursive=True)
        preprocessor = Preprocess(250, 250, peak='R', final_length=500)

        for subject, k in enumerate(list(set([item[:item.rfind('_')] for item in files]))):
            try:
                atrial = pd.read_csv(k + '_AtrialParameters.txt', sep=' ', skiprows=2).ffill(axis=1)
                ventricular = pd.read_csv(k + '_VentricularParameters.txt', sep=' ').ffill(axis=1)
                signal = pd.read_csv(
                    k.replace('WP2_largeDataset_ParameterFiles', 'WP2_largeDataset_Noise') + '_filtered.csv',
                    header=None).loc[0, :]
                data_prep, q, ind = preprocessor.preprocess(data=signal, sampling_rate=500)
                atrial.iloc[:, -1] = atrial.iloc[:, -1].apply(
                    lambda x: x.replace('mm/s', '').replace('deg', '').replace('mm', ''))

                result = [part for part in k.split("/") if part]
                subdiag = len(result) > 14
                for i, t in enumerate(data_prep):
                    dict = {
                        'ecg': {
                            'I': t.flatten(),
                        },
                        'subject': subject,
                        'diagnosis': result[10],
                        'subdiagnosis': result[11] if subdiag else 'None',
                        # TODO: 10 and 11 is set statically --> needs to be adapt
                    }
                    for j in allowed:
                        dict.update({j: 0.0})

                    for j, val in ventricular.iterrows():
                        ke = 'v_' + val.iloc[0]

                        if ke in allowed:
                            va = 0.0
                            try:
                                va = np.float64(val.iloc[-1])
                            except:
                                if (val.iloc[-1] == 'True') | (val.iloc[-1] == 'False'):
                                    va = val.iloc[-1]
                            dict.update({ke: va})

                    for j, val in atrial.iterrows():
                        ke = 'a_' + val.iloc[0]
                        va = 0.0
                        if ke in allowed:
                            try:
                                va = np.float64(val.iloc[-1])
                            except:
                                if (val.iloc[-1] == 'True') | (val.iloc[-1] == 'False'):
                                    va = val.iloc[-1]
                            dict.update({ke: va})

                    yield k + '_' + str(i), dict

            except Exception as e:
                print(e)
                pass
