import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
ds = tfds.load('physionet', split='train', as_supervised=False)

# Define lead names
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Collect up to 100 beats
beats = []

for example in ds:
    ecg_dict = example['ecg']
    try:
        ecg_beat = np.stack([ecg_dict[lead].numpy() for lead in lead_names])  # (12, length)
        beats.append(ecg_beat)
        if len(beats) >= 100:
            break
    except Exception as e:
        print(f"⚠️ Skipping malformed sample: {e}")

# Stack into shape (100, 12, beat_length)
beats = np.stack(beats)
num_beats, num_leads, beat_length = beats.shape

# Create figure with 12 subplots
fig, axes = plt.subplots(12, 1, figsize=(14, 20), sharex=True)
fig.suptitle("Overlay of First 100 ECG Beats with Mean Trace per Lead", fontsize=16, y=1.02)

for i in range(num_leads):
    ax = axes[i]
    for b in range(num_beats):
        ax.plot(beats[b, i], alpha=0.3, linewidth=0.8, color='blue')
    
    # Compute and plot mean beat for this lead
    mean_beat = np.mean(beats[:, i, :], axis=0)
    ax.plot(mean_beat, color='black', linewidth=2.0, label='Mean Beat')

    ax.set_title(f"Lead {lead_names[i]}", loc='left')
    ax.grid(True)
    ax.set_ylabel("mV")
    if i < 11:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("Samples")
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()
