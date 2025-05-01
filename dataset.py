import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, Subset
import numpy as np
import random
import os

# Configurations
DATA_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "data")
NUM_CLIENTS = 10
NON_IID_CLASSES_PER_CLIENT = 2

# Transformations
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_svhn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
])

def load_datasets():
    mnist_train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform_mnist)
    mnist_test = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform_mnist)
    svhn_train = torchvision.datasets.SVHN(root=DATA_DIR, split='train', download=True, transform=transform_svhn)
    svhn_test = torchvision.datasets.SVHN(root=DATA_DIR, split='test', download=True, transform=transform_svhn)
    return mnist_train, mnist_test, svhn_train, svhn_test

def get_targets(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if hasattr(dataset, 'targets'):
        # Convert targets to numpy array without copy parameter
        targets = np.asarray(dataset.targets)
        return targets.astype(np.int64)
    elif hasattr(dataset, 'labels'):
        labels = np.asarray(dataset.labels)
        return labels.astype(np.int64)
    else:
        raise AttributeError("Dataset does not have 'targets' or 'labels' attribute.")

def iid_split(dataset, num_clients):
    data_per_client = len(dataset) // num_clients
    indices = torch.randperm(len(dataset))
    return {i: Subset(dataset, indices[i * data_per_client:(i + 1) * data_per_client]) for i in range(num_clients)}

def noniid_split(dataset, num_clients, classes_per_client):
    full_targets = get_targets(dataset)
    if isinstance(dataset, Subset):
        subset_indices = dataset.indices
        targets = full_targets[subset_indices]
    else:
        targets = full_targets

    class_indices = {cls: np.where(targets == cls)[0] for cls in np.unique(targets)}
    client_indices = {i: [] for i in range(num_clients)}
    available_classes = list(class_indices.keys())
    random.shuffle(available_classes)

    for i in range(num_clients):
        selected_classes = random.sample(available_classes, classes_per_client)
        for cls in selected_classes:
            n_samples = max(1, len(class_indices[cls]) // num_clients)
            chosen = np.random.choice(class_indices[cls], size=n_samples, replace=True)
            client_indices[i].extend(chosen)

    return {i: Subset(dataset, client_indices[i]) for i in range(num_clients)}

def prepare_datasets(dataset, num_clients=NUM_CLIENTS, classes_per_client=NON_IID_CLASSES_PER_CLIENT, iid=True):
    total_indices = list(range(len(dataset)))
    random.shuffle(total_indices)

    test_len = int(0.2 * len(dataset))
    public_len = int(0.1 * (len(dataset) - test_len))
    private_len = len(dataset) - test_len - public_len

    test_indices = total_indices[:test_len]
    public_indices = total_indices[test_len:test_len + public_len]
    private_indices = total_indices[test_len + public_len:]

    test_set = Subset(dataset, test_indices)
    public_set = Subset(dataset, public_indices)
    private_set = Subset(dataset, private_indices)

    if iid:
        clients = iid_split(private_set, num_clients)
    else:
        clients = noniid_split(private_set, num_clients, classes_per_client)

    return {
        "public": public_set,
        "test": test_set,
        "clients": clients
    }