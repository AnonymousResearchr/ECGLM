import torch
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import torch.distributed as dist

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    
    Args:
    - x: Tensor of shape [N, B, D, L], representing the input sequence.
    - mask_ratio: Ratio of the sequence to be masked.

    Returns:
    - x_masked: The masked sequence.
    - mask: The mask applied to the sequence.
    """
    N, B, D, L = x.shape
    len_keep = int(D * (1 - mask_ratio))
    noise = torch.rand(N, B, D, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=2)
    ids_restore = torch.argsort(ids_shuffle, dim=2)
    ids_keep = ids_shuffle[:, :, :len_keep]
    mask = torch.ones([N, B, D], device=x.device)
    mask[:, :, :len_keep] = 0
    mask = torch.gather(mask, dim=2, index=ids_restore)
    x_masked = x.clone()
    x_masked[mask.unsqueeze(-1).repeat(1, 1, 1, L) == 1] = 0
    return x_masked, mask.to(torch.bool)

def tokenize_data_space(tokenizer, data_list, data_list_padded, data_device):
    """
    Tokenizes the data by passing data_list_padded to the space_tokenizer.
    
    Args:
    - tokenizer: The tokenizer to be applied.
    - data_list: List of input data tensors.
    - data_list_padded: List of padded data tensors.
    - data_device: Device to run the tokenizer on.

    Returns:
    - tokenized_data_list: List of tokenized data tensors.
    """
    tokenized_data_list = []
    for i in range(len(data_list)):
        data_padded = data_list_padded[i].to(data_device)
        tokenized_data = tokenizer.tokenize(data_list[i].to(data_device), data_padded)
        tokenized_data_list.append(tokenized_data)
    return tokenized_data_list

def normalize(data, mean, std):
    """
    Normalize the input data with given mean and standard deviation.
    
    Args:
    - data: Input tensor.
    - mean: Mean tensor.
    - std: Standard deviation tensor.

    Returns:
    - Normalized data.
    """
    mean = mean.to(data.device)  # Ensure mean is on the same device
    std = std.to(data.device)    # Ensure std is on the same device
    return (data - mean) / std

def denormalize(data, mean, std):
    """
    Denormalize the input data with given mean and standard deviation.
    
    Args:
    - data: Input tensor.
    - mean: Mean tensor.
    - std: Standard deviation tensor.

    Returns:
    - Denormalized data.
    """
    mean = mean.to(data.device)  # Ensure mean is on the same device
    std = std.to(data.device)    # Ensure std is on the same device
    return data * std + mean

def load_hdf5_file(file_path):
    """
    Loads HDF5 file and returns the dataset.
    
    Args:
    - file_path: Path to the HDF5 file.

    Returns:
    - data: Loaded data as a PyTorch tensor.
    """
    with h5py.File(file_path, 'r') as f:
        data = f['signal'][:]  # Assuming dataset is stored in 'signal' dataset
    return torch.tensor(data, dtype=torch.float32)

def calculate_model_size(model):
    """
    Calculates and prints the size of the model.
    
    Args:
    - model: PyTorch model to calculate the size of.

    Returns:
    - total_params: Total number of parameters in the model.
    - param_size_mb: Size of the model in megabytes.
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    param_size_mb = total_params * 4 / (1024 ** 2)  # Parameter size (MB), assuming each parameter is a float32 (4 bytes)
    if param_size_mb < 1024:
        print(f"Total model size: {param_size_mb:.2f} MB")
    else:
        param_size_gb = param_size_mb / 1024
        print(f"Total model size: {param_size_gb:.2f} GB")
    return total_params, param_size_mb

def load_state_dict_without_module_prefix(state_dict):
    """
    Remove 'module.' prefix from keys in state_dict if present.
    
    Args:
    - state_dict: State dictionary with or without 'module.' prefix.

    Returns:
    - new_state_dict: State dictionary without 'module.' prefix.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def load_file(file_path):
    """
    Loads a file and transfers it to GPU if it has the expected shape.
    
    Args:
    - file_path: Path to the file.

    Returns:
    - data: Loaded data tensor or error message string.
    """
    try:
        data = load_hdf5_file(file_path)
        if data.shape == (5000, 12):
            if torch.isnan(data).any():
                raise ValueError(f"NaN detected in data from {file_path}")
            return data.cuda()  # Load to GPU
    except Exception as e:
        return f"Error loading {file_path}: {e}"
    return None

def process_all_data(split_files, device, log):
    """
    Processes all data files concurrently and logs any errors.
    
    Args:
    - split_files: List of file paths to process.
    - device: Device to transfer the data to.
    - log: Log file object for error logging.

    Returns:
    - all_data: List of processed data tensors.
    """
    all_data = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        future_to_file = {executor.submit(load_file, file_path): file_path for file_path in split_files}
        for future in tqdm(as_completed(future_to_file), total=len(split_files), desc="Loading data", disable=(dist.get_rank() != 0)):
            file_path = future_to_file[future]
            try:
                data = future.result()
                if isinstance(data, str):  # Check if it's an error message
                    print(data)
                    log.write(data + "\n")
                elif data is not None:
                    all_data.append(data)
            except Exception as e:
                error_message = f"Error processing {file_path}: {e}"
                print(error_message)
                log.write(error_message + "\n")

    if all_data:
        print(f'Data type: {all_data[0].dtype}')
        log.write(f'Data type: {all_data[0].dtype}\n')

    return all_data

def process_dataset(all_data, batch_size, device, log):
    """
    Processes the dataset by normalizing it and batching the data.
    
    Args:
    - all_data: List of data tensors.
    - batch_size: Batch size to split the data.
    - device: Device to transfer the data to.
    - log: Log file object for error logging.

    Returns:
    - dataset: List of normalized and batched data tensors.
    """
    num_batches = len(all_data) // batch_size
    remainder = len(all_data) % batch_size

    dataset = [torch.stack(all_data[i * batch_size:(i + 1) * batch_size]).float() for i in range(num_batches)]

    if remainder > 0:
        print('Last batch dropped')
        log.write('Last batch dropped\n')

    all_data = torch.cat(dataset, dim=0).float()
    mean = all_data.mean(dim=(0, 1))
    std = all_data.std(dim=(0, 1))

    dataset = [(normalize(data, mean, std)) for data in dataset]

    if dataset:
        print(f'Data shape: {dataset[0].shape}')
        log.write(f'Data shape: {dataset[0].shape}\n')

    return dataset

def split_dataset(dataset, log):
    """
    Splits the dataset into training and testing sets.
    
    Args:
    - dataset: Dataset to be split.
    - log: Log file object for error logging.

    Returns:
    - train_dataset: Training dataset split.
    - test_dataset: Testing dataset split.
    """
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def tokenize_data_list(data_list, tokenizer, device, desc, log):
    """
    Tokenizes a list of data tensors concurrently.
    
    Args:
    - data_list: List of data tensors.
    - tokenizer: Tokenizer to apply to each data tensor.
    - device: Device to run the tokenizer on.
    - desc: Description for the progress bar.
    - log: Log file object for error logging.

    Returns:
    - tokenized_data_list: List of tokenized data tensors.
    """
    tokenized_data_list = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        future_to_data = {executor.submit(tokenizer.tokenize, data.to(device)): data for data in data_list}
        for future in tqdm(as_completed(future_to_data), desc=desc, total=len(data_list), disable=(dist.get_rank() != 0)):
            data = future_to_data[future]
            try:
                tokenized_data = future.result()
                tokenized_data_list.append(tokenized_data)
            except Exception as e:
                error_message = f"Error tokenizing data {data}: {e}"
                print(error_message)
                log.write(error_message + "\n")
    return tokenized_data_list
