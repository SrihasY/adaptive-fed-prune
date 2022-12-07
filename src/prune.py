import os
import sys
from bisect import bisect

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import cifar_resnet as resnet
import numpy as np
import torch
import torch.nn.functional as F
import torch_pruning as tp
from cifar_resnet import ResNet18
from torchvision import transforms
from torchvision.datasets import CIFAR10

# parser = argparse.ArgumentParser()
# parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
# parser.add_argument('--batch_size', type=int, default=256)
# parser.add_argument('--verbose', action='store_true', default=False)
# parser.add_argument('--total_epochs', type=int, default=100)
# parser.add_argument('--step_size', type=int, default=70)
# parser.add_argument('--round', type=int, default=1)

# args = parser.parse_args()
args = ""

def get_dataloader():
    train_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]), download=True), batch_size=args.batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]), download=True), batch_size=args.batch_size, num_workers=2)
    return train_loader, test_loader


def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred == target).sum()
            total += len(target)
    return correct / total


def train_model(model, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
    for epoch in range(args.total_epochs):
        print("Epoch number " + str(epoch))
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f" % (
                epoch, args.total_epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc = eval(model, test_loader)
        print("Epoch %d/%d, Acc=%.4f" % (epoch, args.total_epochs, acc))
        if best_acc < acc:
            torch.save(model, 'resnet18-round%d.pth' % (args.round))
            best_acc = acc
        scheduler.step()
    print("Best Acc=%.4f" % (best_acc))


def prune_model_with_indices(model, indices=[]):
    if len(indices) == 0:
        return model

    final_prune_indices = {}
    prev_indices = {}
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 3, 32, 32))
    for key in model.state_dict().keys():
        key_list = key.split('.')
        key = key[:-1*(len(key_list[-1])+1)]
        if 'conv' in key or 'shortcut.0' in key:
            final_prune_indices[key + '.in'] = []
            final_prune_indices[key + '.out'] = []
        else:
            final_prune_indices[key] = []
    for key in final_prune_indices.keys():
        prev_indices[key] = []
    def prune_conv(conv, amount=0.2, ids = []):
        plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channel, ids)
        #print(plan)
        for dep, idxs in plan.plan:
            key = None
            if dep.target._name is not None:
                if 'conv' in dep.target._name or 'shortcut.0' in dep.target._name: #conv layer pruning indices
                    if(dep.handler.__class__.__name__ == "ConvOutChannelPruner"):
                        key = dep.target._name + '.out'
                    elif(dep.handler.__class__.__name__ == "ConvInChannelPruner"):
                        key = dep.target._name + '.in'
                elif 'bn' in dep.target._name or 'shortcut.1' in dep.target._name: #batchnorm layer pruning indices
                    if(dep.handler.__class__.__name__ == "BatchnormPruner"):
                        key = dep.target._name
                elif 'linear' in dep.target._name:
                    if(dep.handler.__class__.__name__ == "LinearInChannelPruner"): #fully connected layer pruning indices
                        key = dep.target._name
                        
                if key is not None:
                    inter_prune_ids = list(idxs)
                    for i in reversed(range(len(prev_indices[key]))):
                        temp = []
                        for id in inter_prune_ids:
                            update_id = id+bisect(prev_indices[key][i], id)
                            while update_id in prev_indices[key][i] or update_id in temp:
                                update_id += 1
                            temp.append(update_id)
                        inter_prune_ids = temp
                    inter_prune_ids.sort()
                    prev_indices[key].append(inter_prune_ids)
                    final_prune_indices[key].extend(inter_prune_ids)
                    final_prune_indices[key].sort()
        plan.exec()

    block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
    blk_id = 0
    i = 0
    for m in model.modules():
        if isinstance(m, resnet.BasicBlock):
            prune_conv(m.conv1, block_prune_probs[blk_id], indices[i])
            i = i + 1
            prune_conv(m.conv2, block_prune_probs[blk_id], indices[i])
            i = i + 1
            blk_id += 1
    
    #print(final_prune_indices)
    #remove repeated values, if any
    for key in final_prune_indices.keys():
        layer_indices = list(dict.fromkeys(final_prune_indices[key]))
        #final_prune_indices[key].sort()
        layer_indices.sort()
        final_prune_indices[key] = layer_indices
    return final_prune_indices

def prune_model(model):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 3, 32, 32))
    conv_prune_indices = []
    final_prune_indices = {}
    prev_indices = {}
    for key in model.state_dict().keys():
        key_list = key.split('.')
        key = key[:-1*(len(key_list[-1])+1)]
        if 'conv' in key or 'shortcut.0' in key:
            final_prune_indices[key + '.in'] = []
            final_prune_indices[key + '.out'] = []
        else:
            final_prune_indices[key] = []
            
    for key in final_prune_indices.keys():
        prev_indices[key] = []
    def prune_conv(conv, amount=0.2):
        strategy = tp.strategy.L1Strategy()
        ids = strategy(conv.weight, amount=amount)
        conv_prune_indices.append(np.array(ids))
        plan = DG.get_pruning_plan(conv, tp.prune_conv_out_channel, ids)
        for dep, idxs in plan.plan:
            key = None
            if dep.target._name is not None:
                if 'conv' in dep.target._name or 'shortcut.0' in dep.target._name: #conv layer pruning indices
                    if(dep.handler.__class__.__name__ == "ConvOutChannelPruner"):
                        key = dep.target._name + '.out'
                    elif(dep.handler.__class__.__name__ == "ConvInChannelPruner"):
                        key = dep.target._name + '.in'
                elif 'bn' in dep.target._name or 'shortcut.1' in dep.target._name: #batchnorm layer pruning indices
                    if(dep.handler.__class__.__name__ == "BatchnormPruner"):
                        key = dep.target._name
                elif 'linear' in dep.target._name:
                    if(dep.handler.__class__.__name__ == "LinearInChannelPruner"): #fully connected layer pruning indices
                        key = dep.target._name
                        
                if key is not None:
                    inter_prune_ids = list(idxs)
                    for i in reversed(range(len(prev_indices[key]))):
                        temp = []
                        for id in inter_prune_ids:
                            update_id = id+bisect(prev_indices[key][i], id)
                            while update_id in prev_indices[key][i] or update_id in temp:
                                update_id += 1
                            temp.append(update_id)
                        inter_prune_ids = temp
                    inter_prune_ids.sort()
                    prev_indices[key].append(inter_prune_ids)
                    final_prune_indices[key].extend(inter_prune_ids)
                    final_prune_indices[key].sort()
        plan.exec()

    block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
    block_prune_probs = [0.5*x for x in block_prune_probs]
    blk_id = 0
    for m in model.modules():
        if isinstance(m, resnet.BasicBlock):
            prune_conv(m.conv1, block_prune_probs[blk_id])
            prune_conv(m.conv2, block_prune_probs[blk_id])
            blk_id += 1

    #print(final_prune_indices)
    #remove repeated values, if any
    for key in final_prune_indices.keys():
        layer_indices = list(dict.fromkeys(final_prune_indices[key]))
        layer_indices.sort()
        final_prune_indices[key] = layer_indices
    return conv_prune_indices, final_prune_indices

#python prune_resnet18_cifar10.py --mode prune --round 1 --total_epochs 1 --step_size 20 # 4.5M, Acc=0.9229

def main():
    train_loader, test_loader = get_dataloader()
    if args.mode == 'train':
        args.round = 0
        model = ResNet18(num_classes=10)
        train_model(model, train_loader, test_loader)
    elif args.mode == 'prune':
        previous_ckpt = 'resnet18-round%d.pth' % (args.round - 1)
        print("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt)
        prune_model(model)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        train_model(model, train_loader, test_loader)
    elif args.mode == 'test':
        ckpt = 'resnet18-round%d.pth' % (args.round)
        print("Load model from %s" % (ckpt))
        model = torch.load(ckpt)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n" % (acc))


if __name__ == '__main__':
    main()
