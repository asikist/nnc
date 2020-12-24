from zipfile import ZipFile, ZIP_STORED
import torch
import os

"""
Basic file utilieties for saving tensors and other files in a zip colleciton to reduce file 
clutter from experiments.
"""

def save_tensor_to_collection(collection_path: str, tensor_name: str, tensor: torch.Tensor):
    """
    Appends in or creates a zip file
    :param collection_path: the path to the zip file
    :param tensor_name: The tensor name or path to tensor in the zip
    :param tensor: The tensor.
    :return:
    """
    collection_path = collection_path + ['.zip', ''][collection_path.endswith('.zip')]
    option = ['w', 'a'][os.path.exists(collection_path)]
    tensor_name = tensor_name + ['.pt', ''][tensor_name.endswith('.pt')]

    with ZipFile(collection_path, option, compression=ZIP_STORED) as myzip:
        with myzip.open(tensor_name, 'w') as myfile:
            torch.save(tensor, myfile)


def read_tensor_from_collection(collection_path: str, tensor_name: str, device = 'cpu') -> \
        torch.Tensor:
    """
    Appends in or creates a zip file
    :param collection_path: the path to the zip file
    :param tensor_name: The tensor name or path to tensor in the zip
    :param device: The device to map the tensor. e.g. if tensor was saved directly from cuda,
    you would want to laod it in cpu on a non-cuda machine.
    :return:
    """
    collection_path = collection_path + ['.zip', ''][collection_path.endswith('.zip')]
    tensor_name = tensor_name + ['.pt', ''][tensor_name.endswith('.pt')]
    with ZipFile(collection_path, 'r',  compression=ZIP_STORED) as myzip:
        with myzip.open(tensor_name, 'r') as myfile:
            result = torch.load(myfile, map_location=device)
            return result
