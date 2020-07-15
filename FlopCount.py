#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All rights reserved.
#
'''Train CIFAR10 with PyTorch.'''
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Num epochs=600, lr scheduler after every 100 epochs


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_score,f1_score

import os
import argparse

from src import capsule_model
from utils import progress_bar
import pickle
import json
from datetime import datetime
from utils import seed_torch
from get_dataset import get_dataset

# +
parser = argparse.ArgumentParser(description='Training Capsules using Inverted Dot-Product Attention Routing')

parser.add_argument('--num_routing', default=2, type=int, help='number of routing. Recommended: 0,1,2,3.')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset. CIFAR10,CIFAR100 or MNIST')
parser.add_argument('--backbone', default='resnet', type=str, help='type of backbone. simple or resnet')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers. 0 or 2')
parser.add_argument('--config_path', default='./config_linformer/Local/CIFAR10/resnet_backbone_CIFAR10_capsdim16.json', type=str, help='path of the config')
parser.add_argument('--kernel_transformation', action='store_true', help='not using concurrent_routing')
parser.add_argument('--sequential_routing', action='store_true', help='not using concurrent_routing')
parser.add_argument('--seed', default=12345, type=int, help='Random seed value')
parser.add_argument('--image_dim_size', default=32, type=int, help='image dimension')

parser.add_argument('--dp', default=0.0, type=float, help='dropout rate')
parser.add_argument('--model', default='sinkhorn', type=str, help='default or sinkhorn or bilinear or DynamicBilinear or resnet18')


# -


args = parser.parse_args()
assert args.num_routing > 0
seed_torch(args.seed)
image_dim_size = args.image_dim_size
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


with open(args.config_path, 'rb') as file:
    params = json.load(file)

if args.model=='default':
    net = capsule_model.CapsModel(image_dim_size,
                        params,
                        args.dataset,
                        args.backbone,
                        args.dp,
                        args.num_routing,
                        sequential_routing=args.sequential_routing,
                        seed = args.seed)
elif args.model=='sinkhorn':
    net = capsule_model.CapsSAModel(image_dim_size,
                        params,
                        args.dataset,
                        args.backbone,
                        args.dp,
                        args.num_routing,
                        sequential_routing=args.sequential_routing,
                        seed = args.seed)

elif args.model=='BilinearRandomInit':
    net = capsule_model.CapsRandomInitBAModel(image_dim_size,
                        params,
                        args.dataset,
                        args.backbone,
                        args.dp,
                        args.num_routing,
                        sequential_routing=args.sequential_routing,
                        seed = args.seed)

elif args.model=='bilinear':
    net = capsule_model.CapsBAModel(image_dim_size,
                        params,
                        args.dataset,
                        args.backbone,
                        args.dp,
                        args.num_routing,
                        sequential_routing=args.sequential_routing,
                        seed = args.seed)


if args.model=='LocalLinformer':
    net = capsule_model.CapsBilinearLocalLinformer(image_dim_size,
                        params,
                        args.dataset,
                        args.backbone,
                        args.dp,
                        args.num_routing,
                        kernel_transformation = args.kernel_transformation,
                        sequential_routing=args.sequential_routing,
                        seed = args.seed)

if args.model=='MultipleLocalLinformer':
    net = capsule_model.CapsMultipleTransformationsBilinearLocalLinformer(image_dim_size,
                        params,
                        args.dataset,
                        args.backbone,
                        args.dp,
                        args.num_routing,
                        sequential_routing=args.sequential_routing,
                        seed = args.seed)


if args.model=='GlobalLinformer':
    net = capsule_model.CapsBilinearGlobalLinformerModel(image_dim_size,
                        params,
                        args.dataset,
                        args.backbone,
                        args.dp,
                        args.num_routing,
                        sequential_routing=args.sequential_routing,
                        seed = args.seed)


# -
def count_parameters(model):
    ssum=0
    for name, param in model.named_parameters():
        # if param.requires_grad:
        if param.requires_grad and 'capsule_layer' in name:
            # .numel() returns total number of elements
            print(name, param.numel())
            ssum += param.numel()
    print('Caps sum ', ssum)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(net)
total_params = count_parameters(net)
print("Total model paramters: ",total_params)



# Flop count

inputs = torch.randn(1, 3, image_dim_size, image_dim_size)
from torchprofile import profile_macs
macs = profile_macs(net, inputs)
print(macs)

from thop import profile
macs1, params = profile(net, inputs=(inputs,))
print(macs1)

# with torch.autograd.profiler.profile(use_cuda = True) as prof:
#     out = net(inputs)
# print(prof.key_averages().table(sort_by="cuda_time_total"))


