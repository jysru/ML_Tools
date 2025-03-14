import os
import shutil
import tarfile
import numpy as np


def untar(tar_path: str, extract_to: str = r"C:/temp") -> None:
    # Ensure the extraction directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Untar the file
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_to)
        print(f"Extracted {tar_path} to {extract_to}")


def delete_directory(path: str) -> None:
    # Delete the extracted directory and its contents
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
        print(f"Deleted directory {path} and its contents")


def create_batched_indexes(number: int, batch_size: int) -> list[np.ndarray]:
    # Create an array of indices from 0 to number-1
    indices = np.arange(number).astype(np.int64)

    # Calculate the number of full batches
    num_full_batches = number // batch_size

    # Create the list of batches
    batches = []
    for i in range(num_full_batches):
        start = i * batch_size
        end = start + batch_size
        batches.append(indices[start:end])

    # Add the last batch if there's a remainder
    if number % batch_size != 0:
        batches.append(indices[num_full_batches * batch_size:])

    return batches
