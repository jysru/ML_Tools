import numpy as np
import tensorflow as tf
import torch

from sklearn.model_selection import train_test_split
from typing import Literal, Optional, Union, TypeVar, overload


def tf_get_dataset_number_of_elements(dataset: tf.data.Dataset,) -> int:
    if tf_is_dataset_batched(dataset):
        return dataset.reduce(0, lambda x, batch: x + tf.shape(batch[0])[0]).numpy()
    else:
        return len(list(dataset))
    

def tf_is_dataset_batched(dataset: tf.data.Dataset,) -> bool:
    # Check the element_spec
    if isinstance(dataset.element_spec, tuple):
        # For datasets with tuples of (input, label)
        specs = dataset.element_spec
        return any(spec.shape[0] is None for spec in specs)
    else:
        # For datasets with a single element
        return dataset.element_spec.shape[0] is None
    
    
def torch_get_dataloader_number_of_elements(dataloader: torch.utils.data.DataLoader,) -> int:
    return len(dataloader)


def torch_is_dataloader_batched(dataloader: torch.utils.data.DataLoader,) -> bool:
    return True if dataloader.batch_size > 1 else False


def tf_get_random_element_from_dataset(dataset: tf.data.Dataset,) -> tuple[tf.Tensor,]:
    if tf_is_dataset_batched(dataset):
        return tf_get_random_element_from_batched_dataset(dataset)
    else:
        return tf_get_random_element_from_unbatched_dataset(dataset)
    

def tf_get_random_element_from_batched_dataset(dataset: tf.data.Dataset,) -> tuple[tf.Tensor,]:
    # Randomly select a batch index
    random_batch_idx = tf.random.uniform(shape=[], minval=0, maxval=len(dataset), dtype=tf.int64)
    
    # Skip random_batch_idx-1 batches and take only that specific batch
    for i, batch in enumerate(dataset.skip(random_batch_idx).take(1)):
        batch_size = batch[0].shape[0] if isinstance(batch, tuple) else batch.shape[0]
        
        # Get a random index within the batch
        random_element_idx = tf.random.uniform(shape=[], minval=0, maxval=batch_size, dtype=tf.int64)
        
        # Select the feature and label at the random index within the batch
        if isinstance(batch, tuple):
            random_element = tuple([batch[i][random_element_idx] for i in range(len(batch))])
        else:
            random_element = batch[random_element_idx]
        return random_element
    
    
def tf_get_random_element_from_unbatched_dataset(dataset: tf.data.Dataset,) -> tuple[tf.Tensor,]:
    # Shuffle the dataset with a sufficiently large buffer size
    shuffled_dataset = dataset.shuffle(buffer_size=1000)  # Adjust buffer_size if necessary
    
    # Retrieve the first element from the shuffled dataset
    element = next(iter(shuffled_dataset))
    return element


def torch_get_random_element_from_dataloader(dataloader: torch.utils.data.DataLoader,) -> tuple[torch.Tensor,]:
    if torch_is_dataloader_batched(dataloader):
        return torch_get_random_element_from_batched_dataloader(dataloader)
    else:
        return torch_get_random_element_from_unbatched_dataloader(dataloader)
    

def torch_get_random_element_from_unbatched_dataloader(dataloader: torch.utils.data.DataLoader,) -> tuple[torch.Tensor,]:
    # Shuffle the dataloader
    dataloader.shuffle = True
    
    # Retrieve the first element from the shuffled dataloader
    element = next(iter(dataloader))
    return element


def torch_get_random_element_from_batched_dataloader(dataloader: torch.utils.data.DataLoader,) -> tuple[torch.Tensor,]:
    # Create an iterator
    data_iter = iter(dataloader)

    # Get a random batch
    random_batch = next(data_iter)
    
    # Determine the batch size
    batch_size = random_batch[0].size(0)
    
    # Select a random index within the batch
    random_index = np.random.randint(0, batch_size - 1)

    # Get the random element
    random_element = tuple(tensor[random_index] for tensor in random_batch)
    return random_element


@tf.autograph.experimental.do_not_convert
def tf_make_simple_dataset_from_tensor_slices(
        inputs: np.ndarray,
        outputs: np.ndarray,
        batch_size: int = None,
        invert_inputs_outputs: bool = False,
        verbose: bool = True,
        shuffle: bool = False,
        seed: int = 42,
        ):

    # Compute the sizes for each split
    dataset_size = len(inputs)  # Assuming inputs and outputs have the same first dimension

    # Create TensorFlow datasets from numpy arrays
    if invert_inputs_outputs:
        # Create a TensorFlow dataset from the NumPy arrays
        dataset = tf.data.Dataset.from_tensor_slices((outputs, inputs))
    else:
        # Create a TensorFlow dataset from the NumPy arrays
        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
        
    # Shuffle the dataset with the seed
    if shuffle:
        dataset = dataset.shuffle(buffer_size=dataset_size, seed=seed)

    # Batch the datasets if needed
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    # Print the sizes of each dataset
    if verbose:
        print("Dataset size: ", tf_get_dataset_number_of_elements(dataset))
        
    return dataset


@tf.autograph.experimental.do_not_convert
def tf_make_datasets_from_tensor_slices(
        inputs: np.ndarray,
        outputs: np.ndarray,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        batch_size: int = 32,
        seed: int = 42,
        invert_inputs_outputs: bool = False,
        avoid_leakage: bool = True,
        verbose: bool = True,
        ):
    #### BE SUPER CAREFUL WITH THIS!!!!!!!!
    #### https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets
    #### Otherwise it results in strong data leakage!!!!!!
    
    # Compute the sizes for each split
    dataset_size = len(inputs)  # Assuming inputs and outputs have the same first dimension
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size  # The remaining for test

    # Create TensorFlow datasets from numpy arrays
    if invert_inputs_outputs:
        # Create a TensorFlow dataset from the NumPy arrays
        dataset = tf.data.Dataset.from_tensor_slices((outputs, inputs))
    else:
        # Create a TensorFlow dataset from the NumPy arrays
        dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))

    # Shuffle the dataset with the seed: The `avoid_leakage` flag is CRUCIAL to ensure no data leakage between datasets!!!
    if avoid_leakage:
        shuffled_dataset = dataset.shuffle(buffer_size=dataset_size, seed=seed, reshuffle_each_iteration=False)
    else:
        shuffled_dataset = dataset.shuffle(buffer_size=dataset_size, seed=seed, reshuffle_each_iteration=True)

    # Split the dataset into train, validation, and test sets
    train_dataset = shuffled_dataset.take(train_size)  # First 'train_size' elements
    remaining_dataset = shuffled_dataset.skip(train_size)
    val_dataset = remaining_dataset.take(val_size)  # Next 'val_size' elements
    test_dataset = remaining_dataset.skip(val_size)  # The rest for testing

    # Batch the datasets if needed
    if batch_size is not None:
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

    # Print the sizes of each dataset
    if verbose:
        print("Train size: ", tf_get_dataset_number_of_elements(train_dataset))
        print("Validation size: ", tf_get_dataset_number_of_elements(val_dataset))
        print("Test size: ", tf_get_dataset_number_of_elements(test_dataset))
        
    return train_dataset, val_dataset, test_dataset



@tf.autograph.experimental.do_not_convert
def tf_make_datasets_from_sklearn_arrays(
        inputs: np.ndarray,
        outputs: np.ndarray,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        batch_size: int = 32,
        seed: int = 42,
        invert_inputs_outputs: bool = False,
        verbose: bool = True,
        ):
    
    # Step 1: Shuffle globally before splitting
    indices = np.arange(len(inputs))
    np.random.seed(seed)
    np.random.shuffle(indices)

    inputs = inputs[indices]
    outputs = outputs[indices]

    # Step 2: Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(inputs, outputs, test_size=(1-train_ratio), random_state=seed)
    test_size = 1 - train_ratio - val_ratio
    temp_test_size = test_size / val_ratio
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=temp_test_size, random_state=seed)
    
    # Step 3: Create TensorFlow Datasets from the NumPy arrays
    if invert_inputs_outputs:
        train_dataset = tf.data.Dataset.from_tensor_slices((y_train, X_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((y_val, X_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((y_test, X_test))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Step 4: Batch and shuffle the datasets    
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
    
    # Batch the datasets if needed
    if batch_size is not None:
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

    # Print the sizes of each dataset
    if verbose:
        print("Train size: ", tf_get_dataset_number_of_elements(train_dataset))
        print("Validation size: ", tf_get_dataset_number_of_elements(val_dataset))
        print("Test size: ", tf_get_dataset_number_of_elements(test_dataset))

    return train_dataset, val_dataset, test_dataset


def tf_dataset_to_set(dataset: tf.data.Dataset,) -> set:
    if tf_is_dataset_batched(dataset):
        return tf_batched_dataset_to_set(dataset)
    else:
        return tf_unbatched_dataset_to_set(dataset)


def tf_batched_dataset_to_set(dataset: tf.data.Dataset) -> set:
    elements = set()
    for batch in dataset:
        if isinstance(batch, (tuple, list)):
            # Handle batched dataset with tuples/lists
            inputs, labels = batch
            if isinstance(inputs, tf.Tensor) and isinstance(labels, tf.Tensor):
                # Convert batched tensors
                for inp, lbl in zip(inputs, labels):
                    elements.add((tuple(inp.numpy().flatten()), tuple(lbl.numpy().flatten())))
            else:
                # Handle cases where batch is not in tuple format
                for elem in batch:
                    elements.add(tuple(elem.numpy().flatten()))
        else:
            # Handle unbatched dataset
            if isinstance(batch, tf.Tensor):
                elements.add(tuple(batch.numpy().flatten()))
            else:
                # Handle cases with tuples/lists
                for elem in batch:
                    elements.add(tuple(elem.numpy().flatten()))
    return elements


def tf_unbatched_dataset_to_set(dataset: tf.data.Dataset) -> set:
    elements = set()
    for element in dataset:
        if isinstance(element, (tuple, list)):
            elements.add(tuple([tuple(element[i].numpy().flatten()) for i in range(len(element))]))
        else:
            elements.add(tuple(element.numpy().flatten()))
    return elements


def tf_are_datasets_leaking(datasets: dict) -> bool:
    is_any_dataset_leaking = False
    dataset_names = list(datasets.keys())
    
    for i, name1 in enumerate(dataset_names):
        set1 = tf_dataset_to_set(datasets[name1])
        
        for name2 in dataset_names[i+1:]:
            set2 = tf_dataset_to_set(datasets[name2])
            intersection = set1.intersection(set2)
            
            if intersection:
                print(f"Data leakage detected between {name1} and {name2}: {len(intersection)} common elements found.")
                is_any_dataset_leaking = True
            else:
                print(f"No data leakage detected between {name1} and {name2}.")
                
    return is_any_dataset_leaking



def torch_dataloader_to_set(dataloader: torch.utils.data.DataLoader,) -> set:
    all_elements = set()
    for batch in dataloader:
        # Assuming each batch is a tuple of (inputs, labels)
        inputs, labels = batch
        
        # Convert tensors to tuples for hashability
        inputs_tuple = tuple(tuple(input.cpu().numpy()) for input in inputs)
        labels_tuple = tuple(tuple(label.cpu().numpy()) for label in labels)
        
        # Add to set
        all_elements.update(zip(inputs_tuple, labels_tuple))
    
    return all_elements


def torch_are_dataloaders_leaking(dataloaders: dict) -> bool:
    is_any_dataset_leaking = False
    dataloaders_names = list(dataloaders.keys())
    
    for i, name1 in enumerate(dataloaders_names):
        set1 = torch_dataloader_to_set(dataloaders[name1])
        
        for name2 in dataloaders_names[i+1:]:
            set2 = torch_dataloader_to_set(dataloaders[name2])
            intersection = set1.intersection(set2)
            
            if intersection:
                total_elements = len(set1) + len(set2)
                overlap_percentage = len(intersection) / total_elements * 100
                print(f"Data leakage detected between {name1} and {name2}: {len(intersection)} common elements found.")
                print(f"Percentage of overlapping elements: {overlap_percentage:.2f}%")
                is_any_dataset_leaking = True
            else:
                print(f"No data leakage detected between {name1} and {name2}.")
                
    return is_any_dataset_leaking