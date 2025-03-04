import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load the .mat file
file_path = r"C:\Users\Thomas Kaprielian\Documents\4yp_Data\physionet.org\files\challenge-2021\1.0.3\training\chapman_shaoxing\g11\JS10627.mat"
data = scipy.io.loadmat(file_path)

# Print keys to see structure
print("\n🔍 Available keys in .mat file:")
print(data.keys())
# Define expected lead names
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Extract ECG signals (update key based on Step 1 output)
ecg_data = data['val']  # Common key for ECG data in PhysioNet .mat files

# Check ECG shape
print(f"\n✅ ECG Data Shape: {ecg_data.shape}")

# Ensure the data matches expected lead count
if ecg_data.shape[0] == len(lead_names):
    fig, axes = plt.subplots(12, 1, figsize=(12, 20), sharex=True)

    for i, lead in enumerate(lead_names):
        axes[i].plot(ecg_data[i], label=lead)
        axes[i].set_title(f"Lead {lead}")
        axes[i].legend()
        axes[i].grid()

    plt.xlabel("Time (samples)")
    plt.tight_layout()
    plt.show()
else:
    print(f"⚠️ Unexpected shape! Check .mat file keys. Expected {len(lead_names)} leads, but got {ecg_data.shape[0]}")
