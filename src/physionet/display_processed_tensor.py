import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
dataset_name = "physionet"  # Change this if needed
dataset = tfds.load(dataset_name, split='train')

# Filter dataset for only samples with subject 'HR00005.hea'
filtered_samples = [sample for sample in dataset if sample['subject'].numpy().decode() == 'I0001.hea']

# Define the 12 standard ECG leads
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Check if there are samples for the given subject
if not filtered_samples:
    print("No samples found for subject I0001.hea")
    # Extract and print all unique 'subject' values
    subject_ids = set(sample['subject'].numpy().decode() for sample in dataset)

    # Display the subject IDs
    print("Unique Subjects in the Dataset:")
    for subject in sorted(subject_ids):
        print(subject)
else:
    # Create 12 subplots (2 columns: Individual + Average) with increased height spacing
    fig, axes = plt.subplots(12, 2, figsize=(12, 30))  # Keep width, increase height for better spacing

    # Store all signals for averaging
    ecg_signals = {lead: [] for lead in lead_names}
    nsamp_list = []

    # Plot all ECG segments for each lead
    for sample in filtered_samples:
        nsamp = sample['nsamp'].numpy()  # Number of samples in segment
        nsamp_list.append(nsamp)

        # Extract ECG signals for each lead
        for lead in lead_names:
            ecg_signal = np.array(sample['ecg'][lead])
            ecg_signals[lead].append(ecg_signal[:nsamp])

            # Plot individual lead signals in the left column
            axes[lead_names.index(lead), 0].plot(range(nsamp), ecg_signal[:nsamp], alpha=0.5)

    # Compute the minimum segment length for alignment
    min_nsamp = min(nsamp_list)

    # Compute and plot the average ECG signal for each lead
    for lead in lead_names:
        avg_ecg_signal = np.mean([sig[:min_nsamp] for sig in ecg_signals[lead]], axis=0)

        # Plot the average lead signal in the right column
        axes[lead_names.index(lead), 1].plot(range(min_nsamp), avg_ecg_signal, color='red', linewidth=2)

    # Customize plots
    for i, lead in enumerate(lead_names):
        axes[i, 0].set_title(f"{lead} - Individual Segments")
        axes[i, 0].set_ylabel("ECG Amplitude")
        axes[i, 0].grid()

        axes[i, 1].set_title(f"{lead} - Average Signal")
        axes[i, 1].grid()

    axes[-1, 0].set_xlabel("Time (samples)")
    axes[-1, 1].set_xlabel("Time (samples)")

    # Adjust spacing between subplots (increase vertical space)
    plt.subplots_adjust(hspace=4)  # Increase height spacing

    # Display the plots
    plt.show()
