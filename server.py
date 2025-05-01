import flwr as fl
from flwr.server import Server
from flwr.common import EventType

# Define a custom strategy for the Federated Server
class CustomStrategy(fl.server.strategy.FedAvg):
    def configure_fit(self, rnd, parameters, client_manager):
        print(f"Configuring round {rnd}...")
        return super().configure_fit(rnd, parameters, client_manager)

    def aggregate_fit(self, rnd, results, failures):
        print(f"Aggregating results for round {rnd}...")
        return super().aggregate_fit(rnd, results, failures)

def main() -> Server:
    # Create strategy and server
    strategy = CustomStrategy()
    
    # Create the server with the strategy
    server = Server(strategy=strategy)
    
    # Add event handlers
    def on_server_start(event):
        print("Server started!")
    
    def on_round_start(event):
        print(f"Round {event.round_number} started!")
    
    server.add_event_handler(EventType.SERVER_START, on_server_start)
    server.add_event_handler(EventType.ROUND_START, on_round_start)
    
    return server

if __name__ == "__main__":
    print("\nTo start the server, run this command in the terminal:")
    print("flower-superlink --insecure --fleet-api-address=\"127.0.0.1:9092\" --serverappio-api-address=\"127.0.0.1:9093\"")