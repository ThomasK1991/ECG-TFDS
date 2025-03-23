import wfdb
import matplotlib.pyplot as plt
import os

def plot_12lead_ecg(file_path):
    """
    Load and plot 12-lead ECG from a PhysioNet-style .mat file.
    
    Args:
        file_path (str): Path to the .mat file (without extension).
    """
    # Strip extension if present
    file_path = os.path.splitext(file_path)[0]

    try:
        record = wfdb.rdrecord(file_path)
        ecg_data = record.p_signal.T  # Shape: (num_leads, num_samples)
        leads = record.sig_name
        fs = record.fs
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return

    num_leads = len(leads)
    duration = ecg_data.shape[1] / fs
    time = [i/fs for i in range(ecg_data.shape[1])]

    plt.figure(figsize=(15, 10))
    for i in range(num_leads):
        plt.subplot(num_leads, 1, i+1)
        plt.plot(time, ecg_data[i], linewidth=0.8)
        plt.title(leads[i], loc='left', fontsize=10)
        plt.xticks([])
        plt.yticks([])
        if i == num_leads - 1:
            plt.xticks(fontsize=8)
            plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.suptitle(f"12-lead ECG: {os.path.basename(file_path)}", y=1.02)
    plt.show()

# Example usage:
# Change this to the path of your file (without the .mat extension)
file_path = r"C:\Users\Thomas Kaprielian\Documents\4yp_Data\physionet.org\files\challenge-2021\1.0.3\training\ptb-xl\g21\HR20874.mat"
plot_12lead_ecg(file_path)
