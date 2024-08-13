import os
import h5py
import wfdb
import numpy as np
from scipy.io import loadmat
import asyncio
from concurrent.futures import ProcessPoolExecutor
import time

def convert_dat_to_hdf5(dat_file, output_file):
    """
    Converts a .dat file containing ECG signals to HDF5 format.

    Args:
    - dat_file: Path to the .dat file to be converted.
    - output_file: Path where the output HDF5 file will be saved.
    """
    try:
        record = wfdb.rdrecord(dat_file.replace('.dat', ''), sampfrom=0, physical=False)
        signal = record.d_signal

        # Save the signal in HDF5 format
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('signal', data=signal)

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
        mat_data = loadmat(mat_file)
        signal = mat_data['val']

        # Save the signal in HDF5 format
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('signal', data=signal)
        print(f"Converted {mat_file} to {output_file}")

    except Exception as e:
        print(f"Error processing {mat_file}: {e}")

def process_file(input_file, output_file, convert_function):
    """
    Processes a single file by converting it using the specified function.

    Args:
    - input_file: Path to the input file.
    - output_file: Path to the output HDF5 file.
    - convert_function: Function used to convert the file.
    """
    convert_function(input_file, output_file)

async def process_directory(input_root, output_root, file_extension, convert_function, max_workers=4):
    """
    Processes all files with a given extension in a directory, converting them using a specified function.

    Args:
    - input_root: Directory containing the files to be processed.
    - output_root: Directory where the converted files will be saved.
    - file_extension: The file extension to look for (e.g., '.dat' or '.mat').
    - convert_function: The function to use for converting each file.
    - max_workers: Maximum number of workers to use for parallel processing.
    """
    if not os.path.exists(input_root):
        print(f"Input directory {input_root} does not exist")
        return

    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output directory {output_root}")

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = []
        for root, _, files in os.walk(input_root):
            for file in files:
                if file.endswith(file_extension):
                    input_file = os.path.join(root, file)
                    output_file = os.path.join(output_root, file.replace(file_extension, '.hdf5'))
                    tasks.append(loop.run_in_executor(executor, process_file, input_file, output_file, convert_function))

        await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Sample input and output paths
    ptbxl = os.path.expanduser('~/HOME/ECG/ptb-xl/records500')
    output_root = os.path.expanduser('~/HOME/ECG_HDF5/ptbxl')

    # Adjust the number of workers based on your system's CPU cores
    asyncio.run(process_directory(ptbxl, output_root, '.dat', convert_dat_to_hdf5, max_workers=48))
    # Uncomment the line below to process .mat files
    # asyncio.run(process_directory(wfdblead, output_root, '.mat', convert_mat_to_hdf5, max_workers=48))
