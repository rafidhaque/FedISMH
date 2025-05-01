import flwr as fl
import torch
from torch.utils.data import DataLoader
import argparse

# Import your models and data preparation functions
from model import ModelFactory
from dataset import load_datasets, prepare_datasets

class FedISMHClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        for _ in range(config.get("epochs", 1)):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        accuracy = correct / total
        return float(self.criterion(output, target)), total, {"accuracy": accuracy}

# def main(client_id: int):
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load and prepare datasets
#     mnist_train, mnist_test, _, _ = load_datasets()
#     datasets = prepare_datasets(mnist_train, iid=False)
    
#     # Create client's model
#     factory = ModelFactory()
#     model = factory.create_model(
#         'simple' if client_id % 2 == 0 else 'resnet',
#         in_channels=1,
#         num_classes=10
#     ).to(device)
    
#     # Create data loaders for this client
#     train_loader = DataLoader(datasets['clients'][client_id], batch_size=32, shuffle=True)
#     test_loader = DataLoader(datasets['test'], batch_size=32)
    
#     # Create client and convert to Flower client
#     client = FedISMHClient(model, train_loader, test_loader, device)
    
#     # Start client using the new API
#     fl.client.start_client(
#         server_address="127.0.0.1:9092",
#         client=client.to_client()
#     )

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Start a Flower client")
#     parser.add_argument("--client-id", type=int, required=True, help="Client ID (0-9)")
#     args = parser.parse_args()
    
#     main(args.client_id)

# ...existing code...

def main(node_index: int = 0):
    """Main function that creates and returns a client instance."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare datasets
    mnist_train, mnist_test, _, _ = load_datasets()
    datasets = prepare_datasets(mnist_train, iid=False)
    
    # Create client's model with node_index instead of client_id
    factory = ModelFactory()
    model = factory.create_model(
        'simple' if node_index % 2 == 0 else 'resnet',
        in_channels=1,
        num_classes=10
    ).to(device)
    print(f"Created model for node {node_index}")
    
    # Create data loaders for this client
    train_loader = DataLoader(datasets['clients'][node_index], batch_size=32, shuffle=True)
    test_loader = DataLoader(datasets['test'], batch_size=32)
    
    # Create and return client instance
    return FedISMHClient(model, train_loader, test_loader, device)

# Export the main function for SuperLink/SuperNode
client_fn = main