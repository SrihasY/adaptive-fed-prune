from typing import List, Tuple, Callable, Dict

import flwr as fl
from flwr.common import Metrics

from src.strategy import Struct_Prune_Aggregation


server_pruned_ids = []

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(server_round: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {server_pruned_ids: server_pruned_ids}
        return config

    return fit_config

def get_on_evaluate_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def evaluate_config(server_round: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {server_pruned_ids: server_pruned_ids}
        return config

    return evaluate_config

# Define strategy
strategy = Struct_Prune_Aggregation(evaluate_metrics_aggregation_fn=weighted_average)

# Start Flower server
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    on_fit_config_fn = get_on_fit_config_fn(),
    get_on_evaluate_config_fn = get_on_evaluate_config_fn(),
    strategy=strategy,
)
