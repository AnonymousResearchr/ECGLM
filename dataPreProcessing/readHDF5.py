import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def display_ecg_lead(data, title, sampling_rate, duration=10):
    """
    Display a single ECG lead.

    Args:
    - data: The ECG signal data for one lead.
    - title: The title for the plot (usually the lead name).
    - sampling_rate: The sampling rate of the signal in Hz.
    - duration: The duration of the signal to display in seconds (default is 10 seconds).
    """
    time = np.arange(0, len(data)) / sampling_rate
    plt.plot(time[:sampling_rate * duration], data[:sampling_rate * duration])
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()

def display_hdf5_file(file_path):
    """
    Display ECG data from an HDF5 file.

    Args:
    - file_path: The path to the HDF5 file containing the ECG data.
    """
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Read the dataset
        signal = f['signal'][:]
        
        # Determine sampling rate and number of channels from the signal shape
        sampling_rate = signal.shape[0] // 10  # Assuming the data length is 10 seconds
        num_channels = signal.shape[1]
        
        # Display signal information
        print(f'Sampling Rate: {sampling_rate} Hz')
        print(f'Number of Channels: {num_channels}')
        print(f'Signal Shape: {signal.shape}')
        
        # Lead names
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if num_channels < 12:
            leads = leads[:num_channels]
        
        # Plot the ECG signals
        plt.figure(figsize=(15, 10))
        for i, lead in enumerate(leads):
            plt.subplot(4, 3, i + 1)
            display_ecg_lead(signal[:, i], lead, sampling_rate)
        
        plt.tight_layout()
        plt.show()

        # Save the figure to a file
        plt.savefig(os.path.expanduser('~/HOME/ECGLM/ECG/Fig/SampleECG.png'))

if __name__ == "__main__":
    file_path = os.path.expanduser('~/HOME/ECGLM/ECG/SampleHDF5/40895702.hdf5')
    display_hdf5_file(file_path)
