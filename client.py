import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc1(x)

# Define the Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, client_id, train_data, test_data):
        self.model = model
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_data)
        return self.get_parameters(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_data)
        return float(loss), len(self.test_data), {"accuracy": float(accuracy)}

# Define training and testing functions
def train(model, train_data):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for x, y in train_data:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

def test(model, test_data):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for x, y in test_data:
            outputs = model(x)
            loss += loss_fn(outputs, y).item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    return loss, accuracy

if __name__ == "__main__":
    # Example client setup (replace with actual data and model)
    client_id = 0
    model = SimpleModel()
    train_data = []  # Replace with actual training data
    test_data = []  # Replace with actual test data

    # Start Flower client
    client = FlowerClient(model, client_id, train_data, test_data)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)