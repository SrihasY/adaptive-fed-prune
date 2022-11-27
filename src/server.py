from typing import Callable, Dict, List, Tuple

import flwr as fl
from flwr.common import Metrics, Scalar, Config, ndarray_to_bytes
from strategy import Struct_Prune_Aggregation

import numpy as np

server_prune_ids = [[]]*16


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_on_fit_config_fn() -> Callable[[int], Config]:
    """Return a function which returns training configurations."""

    def fit_config(server_round: int) -> Config:
        """Return a configuration with static batch size and (local) epochs."""
        #server_ndarray = np.ndarray((16,0), buffer=np.array(server_prune_ids))
        config = {"server_prune_ids": ndarray_to_bytes(np.array(server_prune_ids))}
        print("server", config)
        return config

    return fit_config


def get_on_evaluate_config_fn() -> Callable[[int], Config]:
    """Return a function which returns training configurations."""

    def evaluate_config(server_round: int) -> Config:
        """Return a configuration with static batch size and (local) epochs."""
        #server_ndarray = np.ndarray((16,0), buffer=np.array(server_prune_ids))
        config = {"server_prune_ids": ndarray_to_bytes(np.array(server_prune_ids))}
        return config

    return evaluate_config


# Define strategy
strategy = Struct_Prune_Aggregation(on_fit_config_fn=get_on_fit_config_fn(),
                                    on_evaluate_config_fn=get_on_evaluate_config_fn())

# Start Flower server
fl.server.start_server(
    server_address="127.0.0.1:9000",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
