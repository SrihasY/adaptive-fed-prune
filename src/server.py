import argparse
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics, Parameters
from strategy import Struct_Prune_Aggregation

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--tot_clients', type=int)
parser.add_argument('--sample_clients', type=int)
parser.add_argument('--serv_addr', type=str)
parser.add_argument('--init_model', type=str)
parser.add_argument('--server_rounds', type=int)
parser.add_argument('--agg_fraction', type=float)
args = parser.parse_args()

#set initial global model
init_params = None
with open(args.init_model, mode="rb") as init_model_file:
    init_params = Parameters(tensors = [init_model_file.read()], tensor_type='bytes')
    
# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define strategy
strategy = Struct_Prune_Aggregation(evaluate_metrics_aggregation_fn=weighted_average, initial_parameters=init_params,
                                    tot_clients = args.tot_clients, sample_clients = args.sample_clients)

# Start Flower server
fl.server.start_server(
    server_address=args.serv_addr,
    config=fl.server.ServerConfig(num_rounds=args.server_rounds),
    strategy=strategy,
)
