from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from typing import List, Tuple
import numpy as np

def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """Aggregate metrics from multiple clients weighted by number of samples."""
    accuracies = [m[1]["accuracy"] * m[0] for m in metrics]
    examples = [m[0] for m in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=5,  # Minimum number of clients required for training
    min_evaluate_clients=5,  # Minimum number of clients required for evaluation
    min_available_clients=5,  # Minimum number of total clients required
    evaluate_metrics_aggregation_fn=weighted_average,  # Custom metrics aggregation function
)

# Server configuration
server_config = ServerConfig(
    num_rounds=10,  # Number of federated learning rounds
    round_timeout=60,  # Timeout for each round in seconds
)