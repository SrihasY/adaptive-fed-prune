from collections import OrderedDict
from io import BytesIO
import json

from logging import WARNING, log
from bisect import bisect
from typing import Callable, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import re

from flwr.server.strategy import FedAvg

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArray,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.strategy.aggregate import aggregate
import torch

from cifar_resnet import ResNet18
from prune import prune_model_with_indices
from utility import custom_bytes_to_ndarray


class Struct_Prune_Aggregation(FedAvg):

    def __init__(self,
                 on_fit_config_fn: Optional[Callable[[int, List[List[int]]], Dict[str, NDArray]]] = None,
                 on_evaluate_config_fn: Optional[Callable[[int, List[List[int]]], Dict[str, NDArray]]] = None,
                 evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
                 ) -> None:
        super().__init__(evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
        self.server_prune_ids = [[]]*16
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.central_parameters = self.initial_parameters
        self.aggregate_frac = 1
        self.server_net = torch.load('resnet18-round3.pth', map_location="cuda" if torch.cuda.is_available() else "cpu")

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        #update server weights
        self.central_parameters = parameters_to_ndarrays(parameters)
        params_dict = zip(self.server_net.state_dict().keys(), self.central_parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.server_net.load_state_dict(state_dict, strict=True)
        
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round, self.server_prune_ids)
        test = custom_bytes_to_ndarray(config['server_prune_ids']).tolist()
        print("initial server prune ids: ", test)
        fit_ins = FitIns(parameters, config)

        #print(fit_ins)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round, self.server_prune_ids)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

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
        client_parameters = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        server_parameters = self.central_parameters

        client_metrics = [json.loads(res.metrics['prune_indices'].decode('utf-8')) for _, res in results]
        client_conv_metrics = [json.loads(res.metrics['conv_prune_indices'].decode('utf-8')) for _, res in results]
        num_examples = [res.num_examples for _, res in results]

        model_dict = self.server_net.state_dict()
        server_prune_ids = []
        tot_examples = np.sum(num_examples)

        for index, key in enumerate(model_dict):
            key_list = key.split('.')
            key = key[:-1*len(key_list[-1])]
            if re.match("^conv[1-2]+$", key_list[-2]) and key != 'conv1.':
                key = key + "out"
                num_channels = (server_parameters[index]).shape[0]
                cardinalities = []
                for channel_idx in range(num_channels):
                    channel_cardinality = 0
                    for client in client_conv_metrics:
                        prune_ids = client[len(server_prune_ids)]
                        if channel_idx in prune_ids:
                            channel_cardinality += num_examples[client_idx]
                    cardinalities.append(channel_cardinality)
                server_prune_ids.append([channel_idx for channel_idx, x in enumerate(cardinalities) if
                                         x >= self.aggregate_frac * tot_examples])
        print("Printing the aggregated indexes.")
        print(server_prune_ids)
        self.server_prune_ids = server_prune_ids
        final_server_prune_indices = prune_model_with_indices(self.server_net, server_prune_ids)
        pruned_parameters = [val.cpu().numpy() for _, val in self.server_net.state_dict().items()]
        
        for array in pruned_parameters:
            array.fill(0)
    
        for index, key in enumerate(model_dict):
            key_list = key.split('.')
            key = key[:-1*len(key_list[-1])]
            if re.match("^conv[1-2]+$", key_list[-2]) or ('shortcut.0' in key):
                out_key = key + "out"
                in_key = key + "in"
                num_out_channels = (server_parameters[index]).shape[0]
                num_in_channels = (server_parameters[index]).shape[1]
                for out_channel in range(num_out_channels):
                    if out_channel in final_server_prune_indices[out_key]:
                        continue #server dropped output channel
                    else:
                        for in_channel in range(num_in_channels):
                            if in_channel in final_server_prune_indices[in_key]:
                                continue #server dropped input channel
                            else:
                                tot_channel_examples = 0
                                server_out_index, server_in_index = get_index(in_key, out_key, in_channel, out_channel, final_server_prune_indices)
                                # print(server_out_index, server_in_index, num_out_channels, num_in_channels, final_server_prune_indices[out_key], final_server_prune_indices[in_key])
                                for client_idx, client_dict in enumerate(client_metrics):
                                    client_out_index, client_in_index = get_index(in_key, out_key, in_channel, out_channel, client_dict)
                                    #print(client_out_index, client_in_index, num_out_channels, num_in_channels, client_dict[out_key], client_dict[in_key])
                                    if client_out_index == -1 and client_in_index == -1:
                                        # print("test")
                                        continue
                                    pruned_parameters[index][server_out_index][server_in_index] = pruned_parameters[index][server_out_index][server_in_index] \
                                                                                                + num_examples[client_idx]*client_parameters[client_idx][index][client_out_index][client_in_index]
                                    tot_channel_examples += num_examples[client_idx]
                                if not tot_channel_examples == 0:
                                    pruned_parameters[index][server_out_index][server_in_index] = pruned_parameters[index][server_out_index][server_in_index]/tot_channel_examples
            elif 'bn' in key or 'shortcut.1' in key:
                if key_list[-1] != 'num_batches_tracked':
                    key = key[:-1]
                    num_out_channels = (server_parameters[index]).shape[0]
                    for out_channel in range(num_out_channels):
                        if out_channel in final_server_prune_indices[key]:
                            continue #server dropped output channel
                        else:
                            server_channel_id = out_channel - bisect(final_server_prune_indices[key], out_channel)
                            tot_channel_examples = 0
                            for client_idx, client_dict in enumerate(client_metrics):
                                client_channel_id = out_channel - bisect(client_dict[key], out_channel)
                                if out_channel in client_dict[key]:
                                    continue
                                pruned_parameters[index][server_channel_id] = pruned_parameters[index][server_channel_id] \
                                                                                            + num_examples[client_idx]*client_parameters[client_idx][index][client_channel_id]
                                tot_channel_examples += num_examples[client_idx]
                            if tot_channel_examples != 0:
                                pruned_parameters[index][server_channel_id] = pruned_parameters[index][server_channel_id]/tot_channel_examples
                else:
                    tot_channel_examples = 0
                    for client_idx, _ in enumerate(client_metrics):
                        pruned_parameters[index] = pruned_parameters[index] + client_parameters[client_idx][index]
                        tot_channel_examples += num_examples[client_idx]
                    pruned_parameters[index] = pruned_parameters[index]/tot_channel_examples
    
        params_dict = zip(self.server_net.state_dict().keys(), pruned_parameters)
        pruned_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.server_net.load_state_dict(pruned_state_dict, strict=True)
        self.central_parameters = pruned_parameters
                        
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(pruned_parameters), metrics_aggregated # To be changed

    # Might need for client personalized model evaluation.
    # def aggregate_evaluate(self, server_round, results, failures):
    #     # Your implementation here``

def get_index(in_key, out_key, in_channel, out_channel, dict):
    if in_channel in dict[in_key] or out_channel in dict[out_key]:
        return -1, -1
    else:
        in_offset = bisect(dict[in_key], in_channel)
        out_offset = bisect(dict[out_key], out_channel)
        return out_channel - out_offset, in_channel - in_offset