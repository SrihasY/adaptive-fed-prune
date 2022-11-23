import flwr

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.server.strategy.strategy import Strategy

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class Struct_Prune_Aggregation(Strategy):
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        # Your implementation here

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Your implementation here

    def aggregate_evaluate(self, server_round, results, failures):
        # Your implementation here``

    def evaluate(self, parameters):
        # Your implementation here