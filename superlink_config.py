from flwr.server import ServerConfig, ServerAddress
from flwr.common import Parameter
from pathlib import Path

def get_config():
    return {
        "server_config": ServerConfig(
            num_rounds=3,
            round_timeout=None,
        ),
        "client_config": {
            "batch_size": 32,
            "epochs": 1,
            "num_clients": 10,
            "iid": False,
        },
        "client_resources": {
            "module_path": str(Path(__file__).parent / "client.py"),
            "function_name": "main"
        }
    }

config = get_config()  # This will be imported by SuperLink