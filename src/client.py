import warnings
import math
import sys
from collections import OrderedDict
import os
import torch
import flwr as fl
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
from cifar_resnet import ResNet18
import cifar_resnet as resnet

import argparse

import numpy as np 

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

#TODO: Change training data split to be iid.
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
            correct += (pred==target).sum()
            total += len(target)
    return loss/len(testloader.dataset), correct / total

# Load model and data (Resnet18, CIFAR-10)
net = ResNet18(num_classes=10)

trainloader, testloader = get_dataloader()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_model(net, trainloader)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = eval(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
