import flwr

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from flwr.server.strategy import FedAvg

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

class Struct_Prune_Aggregation(FedAvg):

    def __init__ (self, ):
        super(Struct_Prune_Aggregation, self).__init__()
        self.central_parameters = self.initial_parameters
        self.aggregate_frac = 0.3

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.central_parameters = parameters
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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        server_weights = parameters_to_ndarrays(self.central_parameters)
        
        num_examples = [res.num_examples for _, res in results]
        client_metrics = [res.metrics for _,res in results]
        
        tot_examples = np.sum(num_examples)
        
        i = 0
        for layer in server_weights:
            if i%6 is 0: #conv layer weights
                num_channels = layer.shape[0]
                cardinalities = []
                for j in range(num_channels):
                    channel_cardinality = 0
                    for client in fit_metrics:
                        #TODO get client metrics index from weight dict index
                        prune_ids = client_metrics['prune_indices'][i]
                        if j in prune_ids:
                            channel_cardinality += client[0]
                    cardinalities.append(channel_cardinality)
                server_prune_ids = [x for x in cardinalities if x >= self.aggregate_frac*tot_examples]
            #elif i%6 is 1: #batch norm weights
            i+=1
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    # Might need for client personalized model evaluation.
    # def aggregate_evaluate(self, server_round, results, failures):
    #     # Your implementation here``