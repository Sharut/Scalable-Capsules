#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All rights reserved.
#
'''Train CIFAR10 with PyTorch.'''
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Num epochs=600, lr scheduler after every 100 epochs


#  CUDA_VISIBLE_DEVICES=1 python3 main_FMNIST_SVNH.py --config_path ./configs/SVHN/resnet_backbone_SVHN_capsdim121v6.json --model bilinear --dataset SVHN --train_bs 32 --accumulation_steps 4 --seed 0 --step_size 10


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms

import math
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

parser.add_argument('--resume_dir', '-r', default='', type=str, help='dir where we resume from checkpoint')
parser.add_argument('--num_routing', default=2, type=int, help='number of routing. Recommended: 0,1,2,3.')
parser.add_argument('--dataset', default='Expanded_AffNISTv2', type=str, help='dataset. CIFAR10,CIFAR100 or MNIST')
parser.add_argument('--backbone', default='resnet', type=str, help='type of backbone. simple or resnet')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers. 0 or 2')
parser.add_argument('--config_path', default='./configs/resnet_backbone_CIFAR100_capsdim1024.json', type=str, help='path of the config')
parser.add_argument('--debug', action='store_true',
                    help='use debug mode (without saving to a directory)')
parser.add_argument('--sequential_routing', action='store_true', help='not using concurrent_routing')
parser.add_argument('--kernel_transformation', action='store_true', help='tranform each 3*3 to 4 tranformation with local linformer')
parser.add_argument('--multi_transforms', action='store_true', help='tranform 288->128 using this number of matrices ( say 4, then 4 matrices to 32 dimension and then concatenate before attention')


parser.add_argument('--shear', action='store_true', help='shearing while training')
parser.add_argument('--train_bs', default=128, type=int, help='Batch Size for train')
parser.add_argument('--test_bs', default=100, type=int, help='Batch Size for test')
parser.add_argument('--seed', default=12345, type=int, help='Random seed value')



parser.add_argument('--accumulation_steps', default=1, type=float, help='Number of gradeitn accumulation steps')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate: 0.1 for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay: 0.1')
parser.add_argument('--dp', default=0.0, type=float, help='dropout rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--total_epochs', default=400, type=int, help='Total epochs for training')
parser.add_argument('--model', default='sinkhorn', type=str, help='default or sinkhorn or bilinear or DynamicBilinear or resnet18')
parser.add_argument('--optimizer', default='SGD', type=str, help='SGD or Adams')
parser.add_argument('--step_size', default='5', type=int, help='step size')

# parser.add_argument('--save_dir', default='CIFAR10', type=str, help='dir to save results')

# -

def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

args = parser.parse_args()
assert args.num_routing > 0
accumulation_steps=args.accumulation_steps
seed_torch(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
train_translation_rotation_list = [((0,0),0)]#,((0.075,0.075),30),((0.075,0.075),60),((0.075,0.075),90),((0.075,0.075),180)]
test_translation_rotation_list = [((0,0),0)]
if args.shear:
    shear = (0.9,1.1)
else:
    shear=None
if args.dataset =='SVHN':
    train_translation_rotation_list = [((0.2,0.2),0)]#,((0.075,0.075),30),((0.075,0.075),60),((0.075,0.075),90),((0.075,0.075),180)]
    test_translation_rotation_list = [((0,0),0)]

translation,rotation = train_translation_rotation_list[0]
train_desc = '_augment_'+str(translation[0])+'_'+str(translation[1])+'_'+str(rotation)+'_'+str(shear)
trainset, testset, num_class, image_dim_size = get_dataset(args.dataset, args.seed, train_translation_rotation_list, test_translation_rotation_list, shear)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False, num_workers=args.num_workers)
print("Training dataset: ", len(trainloader)," Validation dataset: " , len(testloader))

print('==> Building model..')


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

elif args.model=='HintonDynamic':
    print("Using Sara Sabour's Dynamic Routing")
    assert args.sequential_routing == True
    net = capsule_model.CapsDRModel(image_dim_size,
                        params,
                        args.dataset,
                        args.backbone,
                        args.dp,
                        args.num_routing,
                        sequential_routing=args.sequential_routing,
                        seed = args.seed)

elif args.model=='DynamicBilinear':
    assert args.sequential_routing == True
    net = capsule_model.CapsDBAModel(image_dim_size,
                        params,
                        args.dataset,
                        args.backbone,
                        args.dp,
                        args.num_routing,
                        sequential_routing=args.sequential_routing,
                        seed = args.seed)
    
elif args.model=='MultiHeadBilinear':
    net = capsule_model.CapsMultiHeadBAModel(image_dim_size,
                        params,
                        args.dataset,
                        args.backbone,
                        args.dp,
                        args.num_routing,
                        multi_transforms  = args.multi_transforms,
                        sequential_routing=args.sequential_routing,
                        seed = args.seed)


if args.model=='LocalLinformer':
    net = capsule_model.CapsBilinearLocalLinformer(image_dim_size,
                        params,
                        args.dataset,
                        args.backbone,
                        args.dp,
                        args.num_routing,
                        multi_transforms  = args.multi_transforms,
                        kernel_transformation = args.kernel_transformation,
                        sequential_routing=args.sequential_routing,
                        seed = args.seed)


if args.model=='MultiHeadLocalLinformer':
    net = capsule_model.CapsMultiHeadBilinearLocalLinformer(image_dim_size,
                        params,
                        args.dataset,
                        args.backbone,
                        args.dp,
                        args.num_routing,
                        kernel_transformation = args.kernel_transformation,
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


elif args.model=='resnet18':
    net = torchvision.models.resnet18(pretrained=True) 
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 10)





# +
if(args.optimizer=="SGD" or args.optimizer=="SGD_Cyclic"):
    print("Setting Optimizer to SGD")
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
else:
    print("Changed optimizer to Adams, Learning Rate 0.001")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0, amsgrad=False)

lr_scheduler_name = "MultiStepLR_150_250"
lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=args.gamma)

if args.optimizer =="SGD_Cyclic":
    print("Cyclical LR !!!")
    lr_scheduler_name = "Cyclical"
    step_size = 3*len(trainloader)
    clr = cyclical_lr(step_size, min_lr=0.01, max_lr=0.2)
    lr_decay = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])


elif args.optimizer !="SGD":
    print("Setting LR Decay for Adams")
    gamma = args.gamma
    lr_scheduler_name = "SovNetLambdaLR_" + str(gamma)
    # lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma = gamma)
    lr_decay.LambdaLR(optimizer,lambda epoch: max(1e-3,0.96**epoch))#lambda epoch: 0.5**(epoch // 10))


elif args.optimizer =="SGD":
    print("Setting LR Decay for SGD")
    print("HOLAAAAAAAAA")
    gamma = args.gamma
    step_size = args.step_size
    lr_scheduler_name = "StepLR_steps_"+ str(step_size) + "_gamma_" + str(gamma)
    lr_decay = torch.optim.lr_scheduler.StepLR(optimizer=optimizer , step_size=step_size, gamma = gamma)




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


# Get configuration info
capsdim = args.config_path.split('capsdim')[1].split(".")[0] if 'capsdim' in args.config_path else 'normal'
print(capsdim)
save_dir_name = 'model_' + str(args.model)+ '_dataset_' + str(args.dataset) + '_batch_' +str(args.train_bs)+'_acc_'+str(args.accumulation_steps) +  '_epochs_'+ str(args.total_epochs) + '_optimizer_' +str(args.optimizer) + '_lr_'+str(args.lr)+'_scheduler_' + lr_scheduler_name +'_num_routing_' + str(args.num_routing) + '_backbone_' + args.backbone + '_config_'+capsdim + '_sequential_routing_'+str(args.sequential_routing) + train_desc +'_seed_'+str(args.seed)
print(save_dir_name)

if 'Linformer' in args.model:
    print("Linformer directory it is")
    if not os.path.isdir('results/Linformer/'+args.dataset + '/CapsDim' + str(capsdim)) and not args.debug:
        os.makedirs('results/Linformer/'+args.dataset + '/CapsDim' + str(capsdim))

    store_dir = os.path.join('results/Linformer/'+args.dataset + '/CapsDim' + str(capsdim), save_dir_name)  
    if not os.path.isdir(store_dir) :  
        os.mkdir(store_dir)

else:  
    if not os.path.isdir('results/'+args.dataset + '/CapsDim' + str(capsdim)) and not args.debug:
        os.makedirs('results/'+args.dataset + '/CapsDim' + str(capsdim))

    store_dir = os.path.join('results/'+args.dataset + '/CapsDim' + str(capsdim), save_dir_name)  
    if not os.path.isdir(store_dir) :  
        os.mkdir(store_dir)



# 
net = net.to(device)
if device == 'cuda':
    # Multi GPU Data Parallelization
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

loss_func = nn.CrossEntropyLoss()

if args.resume_dir and not args.debug:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Training
def train(epoch): 
    global accumulation_steps
    if(accumulation_steps!=1):
        print("TRAINING WITH GRADIENT ACCUMULATION")

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        v = net(inputs)
        # print(targets.shape, v.shape)
        loss = loss_func(v, targets)
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx+1) % accumulation_steps == 0: 
            # print("Performed Gradient update")  
            optimizer.step()
            optimizer.zero_grad()

        # optimizer.step()

        train_loss += loss.item()
        _, predicted = v.max(dim=1)    
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            v = net(inputs)
            loss = loss_func(v, targets)
            test_loss += loss.item()
            _, predicted = v.max(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
                'net': net.state_dict(),
                'acc': acc,
                'optimizer' : optimizer.state_dict(),
                'epoch': epoch,
        }
        torch.save(state, os.path.join(store_dir, 'ckpt.pth'))
        best_acc = acc
    return 100.*correct/total

# +
results = {
    'total_params': total_params,
    'args': args,
    'params': params,
    'train_acc': [],
    'test_acc': [],
}




total_epochs = args.total_epochs
if not args.debug:    
    store_file = os.path.join(store_dir, 'debug.dct')


for epoch in range(start_epoch, start_epoch+total_epochs):
    results['train_acc'].append(train(epoch))
    lr_decay.step()
    lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
    print("Current LR ", lr_step)
    results['test_acc'].append(test(epoch))
    print('Best accuracy: ', best_acc)
    print(save_dir_name)
    pickle.dump(results, open(store_file, 'wb'))
# -



    
