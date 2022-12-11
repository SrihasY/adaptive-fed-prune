import argparse
import io
import json
import math
import os
import sys
import warnings

import flwr as fl
import torch
import torch.nn.functional as F
from flwr.common import (Code, EvaluateIns, EvaluateRes, FitIns, FitRes,
                         GetParametersIns, GetParametersRes, Status,
                         ndarrays_to_parameters)
from prune import prune_model
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from utility import custom_ndarray_to_bytes

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--total_epochs', type=int, default=1)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--client_index', type=int, default=0)
parser.add_argument('--num_clients', type=int)
parser.add_argument('--serv_addr', type=str)
parser.add_argument('--prune_factor', type=float)
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
    client_split = torch.utils.data.Subset(trainset, list(range(math.floor(len(trainset)*args.client_index/(args.num_clients)),
                     math.floor(len(trainset)*(args.client_index+1)/(args.num_clients)))))
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
    #print(model)
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


# Load model and data (Resnet18, CIFAR-10)
net = None
trainloader, testloader = get_dataloader()
network_parameter = 1  # to be used for scaling in the array

status = Status(code=Code.OK, message="Success")

# Define Flower client
class FlowerClient(fl.client.Client):
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        global net
        ndarrays = [val.cpu().numpy() for _, val in net.state_dict().items()]
        parameters = ndarrays_to_parameters(ndarrays)
        return GetParametersRes(status=status, parameters=parameters)

    # def set_parameters(self, parameters):
    #     params_dict = zip(net.state_dict().keys(), parameters)
    #     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    #     net.load_state_dict(state_dict, strict=True)

    def fit(self, ins: FitIns) -> FitRes:
        global net
        #deserialize model sent by server
        server_model_bytes = ins.parameters.tensors[0]
        #update client model
        net = torch.load(io.BytesIO(server_model_bytes), map_location="cuda" if torch.cuda.is_available() else "cpu")
        train_model(net, trainloader)
        conv_prune_indices, prune_indices = prune_model(net, args.prune_factor)
        prune_indices = json.dumps(prune_indices).encode('utf-8')
        conv_prune_indices = custom_ndarray_to_bytes(conv_prune_indices)
        pruned_index_dict = {"prune_indices": prune_indices, "conv_prune_indices": conv_prune_indices}
        paramresults = self.get_parameters(GetParametersIns(None))
        fitresults = FitRes(status=paramresults.status, parameters=paramresults.parameters, num_examples=len(trainloader.dataset), metrics=pruned_index_dict)
        return fitresults

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        global net
        #deserialize model sent by server
        server_model_bytes = ins.parameters.tensors[0]
        #update client model
        net = torch.load(io.BytesIO(server_model_bytes), map_location="cuda" if torch.cuda.is_available() else "cpu")      
        #do we train?  
        train_model(net, trainloader)
        loss, accuracy = eval(net, testloader)
        #print(loss, accuracy)
        return EvaluateRes(status=status, loss=loss, num_examples=len(testloader.dataset), metrics={"accuracy": accuracy})


# Start Flower client
fl.client.start_client(
    server_address=args.serv_addr,
    client=FlowerClient(),
)
