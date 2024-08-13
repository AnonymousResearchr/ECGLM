import torch
from utils import random_masking
import sys

def train_one_epoch_ECGLM_tokenizer(model, meta_tokenizer, space_tokenizer, data_loader, loss_function, optimizer, mask_ratio, model_device, data_device, pbar=None):
    """
    Trains the ECGLM model for one epoch with a tokenizer that applies both meta and spatial tokenization.

    Args:
    - model: The main model to be trained.
    - meta_tokenizer: The tokenizer for meta features.
    - space_tokenizer: The tokenizer for spatial features.
    - data_loader: DataLoader providing the training data.
    - loss_function: The loss function used for training.
    - optimizer: The optimizer used for training.
    - mask_ratio: The ratio of the data to be masked.
    - model_device: Device to run the model on.
    - data_device: Device to load the data on.
    - pbar: Progress bar for visualizing training progress (optional).

    Returns:
    - Average loss over the epoch.
    """
    model.eval()
    meta_tokenizer.train()
    space_tokenizer.train()
    total_loss = 0.0

    for data in data_loader:
        data = data.to(data_device)
        data = data.permute(0, 1, 3, 2)
        x_masked, mask = random_masking(data, mask_ratio)

        mask = mask.to(data_device)
        x_masked = x_masked.to(data_device)

        data = data.permute(0, 1, 3, 2).squeeze(0)
        x_masked = x_masked.permute(0, 1, 3, 2).squeeze(0)
        mask = mask.squeeze(0)

        optimizer.zero_grad()
        
        # Tokenization
        tokenized_data_meta = meta_tokenizer(x_masked)
        tokenized_data_space = space_tokenizer(tokenized_data_meta)

        output = model(tokenized_data_space)
        label = space_tokenizer(meta_tokenizer(data))
        
        mask = mask.unsqueeze(1).expand(-1, output.size(1), -1)

        masked_loss = loss_function(output[mask], label[mask])
        unmasked_loss = loss_function(output[~mask], label[~mask])
        loss = masked_loss + unmasked_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(meta_tokenizer.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(space_tokenizer.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()

        if pbar:
            pbar.update(1)

    return total_loss / len(data_loader)

def test_one_epoch_ECGLM_tokenizer(model, meta_tokenizer, space_tokenizer, data_loader, loss_function, mask_ratio, model_device, data_device, pbar=None):
    """
    Tests the ECGLM model for one epoch with a tokenizer that applies both meta and spatial tokenization.

    Args:
    - model: The main model to be tested.
    - meta_tokenizer: The tokenizer for meta features.
    - space_tokenizer: The tokenizer for spatial features.
    - data_loader: DataLoader providing the test data.
    - loss_function: The loss function used for testing.
    - mask_ratio: The ratio of the data to be masked.
    - model_device: Device to run the model on.
    - data_device: Device to load the data on.
    - pbar: Progress bar for visualizing testing progress (optional).

    Returns:
    - Average loss over the epoch.
    """
    model.eval()
    meta_tokenizer.eval()
    space_tokenizer.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(data_device)
            data = data.permute(0, 1, 3, 2)
            x_masked, mask = random_masking(data, mask_ratio)

            mask = mask.to(data_device)
            x_masked = x_masked.to(data_device)

            data = data.permute(0, 1, 3, 2).squeeze(0)
            x_masked = x_masked.permute(0, 1, 3, 2).squeeze(0)
            mask = mask.squeeze(0)

            # Tokenization
            tokenized_data_meta = meta_tokenizer(x_masked)
            tokenized_data_space = space_tokenizer(tokenized_data_meta)

            output = model(tokenized_data_space)
            label = space_tokenizer(meta_tokenizer(data))
            
            mask = mask.unsqueeze(1).expand(-1, output.size(1), -1)

            masked_loss = loss_function(output[mask], label[mask])
            unmasked_loss = loss_function(output[~mask], label[~mask])
            loss = masked_loss + unmasked_loss

            if torch.isnan(loss):
                sys.exit("NaN detected in test loss.")

            total_loss += loss.item()

            if pbar:
                pbar.update(1)

    return total_loss / len(data_loader)


def train_one_epoch_ECGLM_model(model, assoc_tokenizer, devia_tokenizer, data_loader, loss_function, optimizer, mask_ratio, model_device, data_device, pbar=None):
    """
    Trains the ECGLM model for one epoch using associated and deviation tokenizers.

    Args:
    - model: The main model to be trained.
    - assoc_tokenizer: The associated tokenizer.
    - devia_tokenizer: The deviation tokenizer.
    - data_loader: DataLoader providing the training data.
    - loss_function: The loss function used for training.
    - optimizer: The optimizer used for training.
    - mask_ratio: The ratio of the data to be masked.
    - model_device: Device to run the model on.
    - data_device: Device to load the data on.
    - pbar: Progress bar for visualizing training progress (optional).

    Returns:
    - Average loss over the epoch.
    """
    model.train()
    assoc_tokenizer.eval()
    devia_tokenizer.eval()
    total_loss = 0.0

    for data in data_loader:
        data = data.to(data_device)
        data = data.permute(0, 1, 3, 2)
        x_masked, mask = random_masking(data, mask_ratio)

        mask = mask.to(data_device)
        x_masked = x_masked.to(data_device)

        data = data.permute(0, 1, 3, 2).squeeze(0)
        x_masked = x_masked.permute(0, 1, 3, 2).squeeze(0)
        mask = mask.squeeze(0)

        optimizer.zero_grad()

        # Tokenization
        tokenized_data_meta = assoc_tokenizer(x_masked)
        tokenized_data_space = devia_tokenizer(tokenized_data_meta)

        output = model(tokenized_data_space)
        label = devia_tokenizer(assoc_tokenizer(data))

        mask = mask.unsqueeze(1).expand(-1, output.size(1), -1)

        masked_loss = loss_function(output[mask], label[mask])
        unmasked_loss = loss_function(output[~mask], label[~mask])
        loss = masked_loss + unmasked_loss

        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if pbar:
            pbar.update(1)

    return total_loss / len(data_loader)


def test_one_epoch_ECGLM_model(model, assoc_tokenizer, devia_tokenizer, data_loader, loss_function, mask_ratio, model_device, data_device, pbar=None):
    """
    Tests the ECGLM model for one epoch using associated and deviation tokenizers.

    Args:
    - model: The main model to be tested.
    - assoc_tokenizer: The associated tokenizer.
    - devia_tokenizer: The deviation tokenizer.
    - data_loader: DataLoader providing the test data.
    - loss_function: The loss function used for testing.
    - mask_ratio: The ratio of the data to be masked.
    - model_device: Device to run the model on.
    - data_device: Device to load the data on.
    - pbar: Progress bar for visualizing testing progress (optional).

    Returns:
    - Average loss over the epoch.
    """
    model.eval()
    assoc_tokenizer.eval()
    devia_tokenizer.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(data_device)
            data = data.permute(0, 1, 3, 2)
            x_masked, mask = random_masking(data, mask_ratio)

            mask = mask.to(data_device)
            x_masked = x_masked.to(data_device)

            data = data.permute(0, 1, 3, 2).squeeze(0)
            x_masked = x_masked.permute(0, 1, 3, 2).squeeze(0)
            mask = mask.squeeze(0)

            # Tokenization
            tokenized_data_meta = assoc_tokenizer(x_masked)
            tokenized_data_space = devia_tokenizer(tokenized_data_meta)

            output = model(tokenized_data_space)
            label = devia_tokenizer(assoc_tokenizer(data))

            mask = mask.unsqueeze(1).expand(-1, output.size(1), -1)

            masked_loss = loss_function(output[mask], label[mask])
            unmasked_loss = loss_function(output[~mask], label[~mask])
            loss = masked_loss + unmasked_loss

            if torch.isnan(loss):
                sys.exit("NaN detected in test loss.")

            total_loss += loss.item()

            if pbar:
                pbar.update(1)

    return total_loss / len(data_loader)


def train_one_epoch_simple(model, tokenizer, data_loader, loss_function, optimizer, mask_ratio, model_device, data_device, pbar=None):
    """
    Trains the model for one epoch using a simple tokenizer.

    Args:
    - model: The main model to be trained.
    - tokenizer: The tokenizer to be applied to the input data.
    - data_loader: DataLoader providing the training data.
    - loss_function: The loss function used for training.
    - optimizer: The optimizer used for training.
    - mask_ratio: The ratio of the data to be masked.
    - model_device: Device to run the model on.
    - data_device: Device to load the data on.
    - pbar: Progress bar for visualizing training progress (optional).

    Returns:
    - Average loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for data in data_loader:
        data = data.to(data_device)
        data = data.permute(0, 1, 3, 2)
        x_masked, mask = random_masking(data, mask_ratio)

        x_masked = x_masked.to(model_device)
        data = data.permute(0, 1, 3, 2).squeeze(0)
        x_masked = x_masked.permute(0, 1, 3, 2).squeeze(0)
        mask = mask.squeeze(0)

        data = data.to(data_device)
        mask = mask.to(data_device)
        x_masked = x_masked.to(data_device)

        optimizer.zero_grad()
        
        tokenized_data = x_masked
        output = model(tokenized_data)
        label = data
        
        mask = mask.unsqueeze(1).expand(-1, output.size(1), -1)
        masked_loss = loss_function(output[mask], label[mask])
        unmasked_loss = loss_function(output[~mask], label[~mask])
        total_loss = masked_loss + unmasked_loss

        total_loss.backward()
        optimizer.step()
        
        total_loss += total_loss.item()

        if pbar:
            pbar.update(1)
    
    return total_loss / len(data_loader)

def test_one_epoch_simple(model, tokenizer, data_loader, loss_function, mask_ratio, model_device, data_device, pbar=None):
    """
    Tests the model for one epoch using a simple tokenizer.

    Args:
    - model: The main model to be tested.
    - tokenizer: The tokenizer to be applied to the input data.
    - data_loader: DataLoader providing the test data.
    - loss_function: The loss function used for testing.
    - mask_ratio: The ratio of the data to be masked.
    - model_device: Device to run the model on.
    - data_device: Device to load the data on.
    - pbar: Progress bar for visualizing testing progress (optional).

    Returns:
    - Average loss over the epoch.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(data_device)
            data = data.permute(0, 1, 3, 2)
            x_masked, mask = random_masking(data, mask_ratio)

            x_masked = x_masked.to(model_device)
            data = data.permute(0, 1, 3, 2).squeeze(0)
            x_masked = x_masked.permute(0, 1, 3, 2).squeeze(0)
            mask = mask.squeeze(0)

            data = data.to(data_device)
            mask = mask.to(data_device)
            x_masked = x_masked.to(data_device)

            tokenized_data = x_masked
            output = model(tokenized_data)
            label = data

            mask = mask.unsqueeze(1).expand(-1, output.size(1), -1)
            masked_loss = loss_function(output[mask], label[mask])
            unmasked_loss = loss_function(output[~mask], label[~mask])
            total_loss = masked_loss + unmasked_loss

            if torch.isnan(total_loss):
                sys.exit("NaN detected in test loss.")

            total_loss += total_loss.item()

            if pbar:
                pbar.update(1)

    return total_loss / len(data_loader)

def train_one_epoch_CROSS(model, tokenizer, data_loader, loss_function, optimizer, data_device, pbar=None):
    """
    Trains the model for one epoch using a cross-tokenizer approach.

    Args:
    - model: The main model to be trained.
    - tokenizer: The cross-tokenizer to be applied to the input data.
    - data_loader: DataLoader providing the training data.
    - loss_function: The loss function used for training.
    - optimizer: The optimizer used for training.
    - data_device: Device to load the data on.
    - pbar: Progress bar for visualizing training progress (optional).

    Returns:
    - Average loss over the epoch.
    """
    model.train()
    tokenizer.train()
    total_loss = 0.0

    for data in data_loader:
        data = data.to(data_device)

        mean_d = data.mean(dim=1, keepdim=True)
        std_d = data.std(dim=1, keepdim=True)
        data = (data - mean_d) / (std_d + 1e-8)
        
        if torch.isnan(data).any():
            sys.exit("NaN detected in data after normalization during training.")

        optimizer.zero_grad()
        
        # Tokenization
        tokenized_data = tokenizer(data)
        
        if torch.isnan(tokenized_data).any():
            sys.exit("NaN detected in tokenized data during training.")

        output = model(tokenized_data)
        
        if torch.isnan(output).any():
            sys.exit("NaN detected in model output during training.")

        loss = loss_function(output, data)
        
        if torch.isnan(loss):
            sys.exit("NaN detected in training loss.")

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=1.0)

        optimizer.step()
        
        total_loss += loss.item()

        if pbar:
            pbar.update(1)

    return total_loss / len(data_loader)

def test_one_epoch_CROSS(model, tokenizer, data_loader, loss_function, data_device, pbar=None):
    """
    Tests the model for one epoch using a cross-tokenizer approach.

    Args:
    - model: The main model to be tested.
    - tokenizer: The cross-tokenizer to be applied to the input data.
    - data_loader: DataLoader providing the test data.
    - loss_function: The loss function used for testing.
    - data_device: Device to load the data on.
    - pbar: Progress bar for visualizing testing progress (optional).

    Returns:
    - Average loss over the epoch.
    """
    model.eval()
    tokenizer.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(data_device)
            
            mean_d = data.mean(dim=1, keepdim=True)
            std_d = data.std(dim=1, keepdim=True)
            data = (data - mean_d) / (std_d + 1e-8)
            
            if torch.isnan(data).any():
                sys.exit("NaN detected in data after normalization during testing.")

            # Tokenization
            tokenized_data = tokenizer(data)
            
            if torch.isnan(tokenized_data).any():
                sys.exit("NaN detected in tokenized data during testing.")

            output = model(tokenized_data)
            
            if torch.isnan(output).any():
                sys.exit("NaN detected in model output during testing.")

            loss = loss_function(output, data)

            if torch.isnan(loss):
                sys.exit("NaN detected in test loss.")

            total_loss += loss.item()

            if pbar:
                pbar.update(1)

    return total_loss / len(data_loader)

def train_one_epoch_DEVIAT(model, tokenizer, data_loader, loss_function, optimizer, data_device, codebook, pbar=None):
    """
    Trains the model for one epoch using a deviation tokenizer with a codebook.

    Args:
    - model: The main model to be trained.
    - tokenizer: The deviation tokenizer to be applied to the input data.
    - data_loader: DataLoader providing the training data.
    - loss_function: The loss function used for training.
    - optimizer: The optimizer used for training.
    - data_device: Device to load the data on.
    - codebook: Codebook used in the deviation tokenizer.
    - pbar: Progress bar for visualizing training progress (optional).

    Returns:
    - Average loss over the epoch.
    """
    model.train()
    tokenizer.train()
    total_loss = 0.0

    for data in data_loader:
        data = data.to(data_device)

        mean_d = data.mean(dim=1, keepdim=True)
        std_d = data.std(dim=1, keepdim=True)
        data = (data - mean_d) / (std_d + 1e-8)
        
        if torch.isnan(data).any():
            sys.exit("NaN detected in data after normalization during training.")

        optimizer.zero_grad()
        
        # Tokenization with codebook
        tokenized_data = tokenizer(data, codebook)
        
        if torch.isnan(tokenized_data).any():
            sys.exit("NaN detected in tokenized data during training.")

        output = model(tokenized_data)
        
        if torch.isnan(output).any():
            sys.exit("NaN detected in model output during training.")

        loss = loss_function(output, data)
        
        if torch.isnan(loss):
            sys.exit("NaN detected in training loss.")

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=1.0)

        optimizer.step()
        
        total_loss += loss.item()

        if pbar:
            pbar.update(1)

    return total_loss / len(data_loader)

def test_one_epoch_DEVIAT(model, tokenizer, data_loader, loss_function, data_device, codebook, pbar=None):
    """
    Tests the model for one epoch using a deviation tokenizer with a codebook.

    Args:
    - model: The main model to be tested.
    - tokenizer: The deviation tokenizer to be applied to the input data.
    - data_loader: DataLoader providing the test data.
    - loss_function: The loss function used for testing.
    - data_device: Device to load the data on.
    - codebook: Codebook used in the deviation tokenizer.
    - pbar: Progress bar for visualizing testing progress (optional).

    Returns:
    - Average loss over the epoch.
    """
    model.eval()
    tokenizer.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(data_device)
            
            mean_d = data.mean(dim=1, keepdim=True)
            std_d = data.std(dim=1, keepdim=True)
            data = (data - mean_d) / (std_d + 1e-8)
            
            if torch.isnan(data).any():
                sys.exit("NaN detected in data after normalization during testing.")

            # Tokenization with codebook
            tokenized_data = tokenizer(data, codebook)
            
            if torch.isnan(tokenized_data).any():
                sys.exit("NaN detected in tokenized data during testing.")

            output = model(tokenized_data)
            
            if torch.isnan(output).any():
                sys.exit("NaN detected in model output during testing.")

            loss = loss_function(output, data)

            if torch.isnan(loss):
                sys.exit("NaN detected in test loss.")

            total_loss += loss.item()

            if pbar:
                pbar.update(1)

    return total_loss / len(data_loader)
