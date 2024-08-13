import wfdb
import h5py
import os
import numpy as np
from scipy.io import loadmat

def convert_dat_to_hdf5(dat_file, output_file):
    """
    Converts a .dat file containing ECG signals to HDF5 format.

    Args:
    - dat_file: Path to the .dat file to be converted.
    - output_file: Path where the output HDF5 file will be saved.
    """
    try:
        # Read the ECG signal from the .dat file
        record = wfdb.rdrecord(dat_file.replace('.dat', ''), sampfrom=0, physical=False)
        signal = record.d_signal

        # Ensure the signal has the expected shape [5000, 12]
        if signal.shape[0] != 5000 or signal.shape[1] != 12:
            raise ValueError(f"Signal shape is {signal.shape}, expected [5000, 12]")

        # Save the signal in HDF5 format
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('signal', data=signal)
        print(f"Converted {dat_file} to {output_file}")

    except Exception as e:
        print(f"Error processing {dat_file}: {e}")

def convert_mat_to_hdf5(mat_file, output_file):
    """
    Converts a .mat file containing ECG signals to HDF5 format.

    Args:
    - mat_file: Path to the .mat file to be converted.
    - output_file: Path where the output HDF5 file will be saved.
    """
    try:
        # Read the ECG data from the .mat file
        mat_data = loadmat(mat_file)
        signal = mat_data['val']

        # Ensure the signal has the expected shape [5000, 12]
        if signal.shape[0] != 5000 or signal.shape[1] != 12:
            raise ValueError(f"Signal shape is {signal.shape}, expected [5000, 12]")

        # Save the signal in HDF5 format
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('signal', data=signal)
        print(f"Converted {mat_file} to {output_file}")

    except Exception as e:
        print(f"Error processing {mat_file}: {e}")

def process_directory(input_root, output_root, file_extension, convert_function):
    """
    Processes all files with a given extension in a directory, converting them using a specified function.

    Args:
    - input_root: Directory containing the files to be processed.
    - output_root: Directory where the converted files will be saved.
    - file_extension: The file extension to look for (e.g., '.dat' or '.mat').
    - convert_function: The function to use for converting each file.
    """
    if not os.path.exists(input_root):
        print(f"Input directory {input_root} does not exist")
        return

    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output directory {output_root}")

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith(file_extension):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_root, file.replace(file_extension, '.hdf5'))
                print(f"Processing {input_file}")
                convert_function(input_file, output_file)

if __name__ == "__main__":
    sampleECG = os.path.expanduser('~/HOME/ECGLLM/ECG/ECGSample')
    output_root = os.path.expanduser('~/HOME/ECGLLM/ECG/SampleHDF5')

    process_directory(sampleECG, output_root, '.dat', convert_dat_to_hdf5)
