import argparse
import json
import math
import os
import sys
import warnings
from collections import OrderedDict

from io import BytesIO

import torch
import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F
from flwr.common import bytes_to_ndarray, ndarray_to_bytes, NDArray
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from cifar_resnet import ResNet18
from prune import prune_model, prune_model_with_indices

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--total_epochs', type=int, default=1)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--client_index', type=int, default=0)
parser.add_argument('--num_clients', type=int)
args = parser.parse_args()

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: Change training data split to be iid.
def get_dataloader():
    trainset = CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]), download=True)
    client_split = torch.utils.data.Subset(trainset, list(range(math.floor(len(trainset)*args.client_index/(10*args.num_clients)),
                     math.floor(len(trainset)*(args.client_index+1)/(10*args.num_clients)))))
    train_loader = torch.utils.data.DataLoader(client_split,
                        batch_size=args.batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]),download=True),batch_size=args.batch_size, num_workers=2)
    return train_loader, test_loader


def train_model(model, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)

    for epoch in range(args.total_epochs):
        model.train()
        for img, target in tqdm(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
        scheduler.step()


def eval(model, test_loader):
    correct = 0
    total = 0
    loss = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for img, target in tqdm(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            loss += F.cross_entropy(out, target).item()
            target = target.cpu().numpy()
            correct += (pred == target).sum()
            total += len(target)
    return loss / len(testloader.dataset), correct / total


def custom_ndarray_to_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(bytes_io, ndarray, allow_pickle=True)  # type: ignore
    return bytes_io.getvalue()


# Load model and data (Resnet18, CIFAR-10)
central_net = ResNet18(num_classes=10)
net = central_net
trainloader, testloader = get_dataloader()
network_parameter = 1  # to be used for scaling in the array

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(central_net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        central_net.load_state_dict(state_dict, strict=True)

    #TODO: parameters are being updated twice
    def fit(self, parameters, config):
        #print("test")
        server_prune_ids = bytes_to_ndarray(config['server_prune_ids']).tolist()
        print(server_prune_ids)
        prune_model_with_indices(central_net, server_prune_ids)
        self.set_parameters(parameters)
        net = central_net
        train_model(net, trainloader)
        prune_indices = prune_model(net)
        prune_indices = json.dumps(prune_indices).encode('utf-8')
        pruned_index_dict = {"prune_indices": prune_indices}
        return self.get_parameters(config={}), len(trainloader.dataset), pruned_index_dict

    def evaluate(self, parameters, config):
        server_prune_ids = bytes_to_ndarray(config['server_prune_ids'])
        prune_model_with_indices(central_net, server_prune_ids)
        self.set_parameters(parameters)
        loss, accuracy = eval(central_net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:9000",
    client=FlowerClient(),
)
