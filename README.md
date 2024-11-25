# ECG-TFDS

Load common open source ECG databases as a Tensorflow dataset. The final format should be ECG signals of one second in length, representing a heartbeat. The R-peaks form the center of the signal, which is normalized in the range $\[0, 1\]$.

## How to get started?

1. Install the requirements
```pip install -r requirements```
2. Navigate into the desired dataset folder
```cd src/dataset/```
3. Build the dataset
```tfds build```

## Available Datasets
1. Shaoxing (zheng): https://www.nature.com/articles/s41597-020-0386-x
2. PTB-XL: https://www.nature.com/articles/s41597-020-0495-6
3. Icentia11k: https://physionet.org/content/icentia11k-continuous-ecg/1.0/
4. MedalCare-XL: https://www.nature.com/articles/s41597-023-02416-4
5. ECGSYN: https://physionet.org/content/ecgsyn/1.0.0/
6. Custom: [Add a custom dataset](https://github.com/CardioKit/ECG-TFDS/blob/main/src/custom/README.md)

## How to Add a New Dataset?

To incorporate a new dataset into the collection, follow these steps:

1. Choose an open-source ECG database, e.g., [PhysioNet](https://physionet.org/about/database/). Ensure that you exclusively select datasets with an appropriate license.
2. Go to the `src` folder and execute the following command:
   ```sh
   tfds new DATASET_NAME
3. Open and edit the `DATASET_NAME/DATASET_NAME_dataset_builder.py` file following the provided instructions. Any generic preprocessing steps should be placed in the `utils` folder.
4. Include tests in the `DATASET_NAME/DATASET_NAME_dataset_builder_test.py` file.
5. Include metadata files: `CITATIONS.bib`, `README.md`, and `TAGS.txt`.
6. Confirm successful dataset building using the command:
   ```sh
   tfds build
7. During your initial complete build, register the checksum with:
   ```sh
   tfds build --register_checksums
8. Navigate to `./electrocardiogram` and append the dataset to the collection in `electrocardiogram.py`.
9. Modify the `requirements.txt` and this `README` file accordingly
10. Create a pull request and provide a concise motivation, description, and dataset metadata, including details like count, size, dataset license, and source.

Ensure adherence to these steps to seamlessly integrate the new dataset into the collection.

## How to cite?

If you are using this repository please reference the following article: (to be determined)

```
@article{kapsecker2025disentangled,
  title={Disentangled representational learning for anomaly detection in single-lead electrocardiogram signals using variational autoencoder},
  author={Kapsecker, Maximilian and Möller, Matthias C and Jonas, Stephan M},
  journal={Computers in Biology and Medicine},
  volume={184},
  pages = {109422},
  year = {2025},
  issn = {0010-4825},
  doi = {https://doi.org/10.1016/j.compbiomed.2024.109422},
  url = {https://www.sciencedirect.com/science/article/pii/S0010482524015075},
}
```

