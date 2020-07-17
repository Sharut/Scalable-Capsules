import argparse
import os
import random
import shutil
import time
import warnings

import torch    
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from src import capsule_model
from utils import progress_bar
import pickle
import json
from datetime import datetime
from utils import seed_torch
# os.environ["CUDA_DEVICE_ORDER"]=PCI_BUS_ID

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Training Capsules using Inverted Dot-Product Attention Routing')
parser.add_argument('--dataset', default='ImageNet', type=str, help='dataset. CIFAR10 or CIFAR100.')
parser.add_argument('--resume_dir', '-r', default='', type=str, help='dir where we resume from checkpoint')
parser.add_argument('--num_routing', default=1, type=int, help='number of routing. Recommended: 0,1,2,3.')
parser.add_argument('--backbone', default='resnet', type=str, help='type of backbone. simple or resnet')
parser.add_argument('--config_path', default='./configs/ImageNet.json', type=str, help='path of the config')
parser.add_argument('--debug', action='store_true',
					help='use debug mode (without saving to a directory)')
parser.add_argument('--sequential_routing', action='store_true', help='not using concurrent_routing')
parser.add_argument('--kernel_transformation', action='store_true', help='tranform each 3*3 to 4 tranformation with local linformer')
parser.add_argument('--multi_transforms', action='store_true', help='tranform 288->128 using this number of matrices ( say 4, then 4 matrices to 32 dimension and then concatenate before attention')


parser.add_argument('--dp', default=0.0, type=float, help='dropout rate')
parser.add_argument('--model', default='sinkhorn', type=str, help='default or sinkhorn')


parser.add_argument('--seed', default=0, type=int, help='Random seed value')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')

parser.add_argument('-acc', '--accumulation-steps', default=8, type=int,
					metavar='N',
					help='Gradient accumulation steps')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=32, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
					help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
					help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
					help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
					help='distributed backend')
parser.add_argument('--seed', default=12345, type=int,
					help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
					help='Use multi-processing distributed training to launch '
						 'N processes per node, which has N GPUs. This is the '
						 'fastest way to use PyTorch for either single node or '
						 'multi node data parallel training')


best_acc1 = 0
args = parser.parse_args()
assert args.num_routing > 0
arg_filename = args.config_path.split("/configs/")[1].split(".")[0]
store_dir_savename = 'Baseline_model_' + str(args.model)+ '_dataset_ImageNet' + '_batch_'+ str(args.batch_size) + '_accumulation_'+ str(args.accumulation_steps) + '_epochs_'+ str(args.epochs)+'_num_routing_' + str(args.num_routing) + '_backbone_' + args.backbone + '_config_'+arg_filename
if not os.path.isdir('results'):
	os.mkdir('results')

store_dir = os.path.join('results', store_dir_savename)  
if not os.path.isdir(store_dir):
	os.mkdir(store_dir)

accumulation_steps = args.accumulation_steps

def main():
	global store_dir
	global accumulation_steps
	print(store_dir)

	args = parser.parse_args()


	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
					  'disable data parallelism.')

	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])

	# World size is number of GPU servers/nodes
	args.distributed = args.world_size > 1 or args.multiprocessing_distributed
	

	# By default goes to all 7 GPU's
	ngpus_per_node = torch.cuda.device_count()
	ngpus_per_node=3
	print("Number of GPUs per node-", ngpus_per_node)

	
	if args.multiprocessing_distributed:
		print("Multi-threading Spawning...")
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed processes: the
		# main_worker process function
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		print("No multi-threading")
		# Simply call main_worker function
		main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

	global store_dir, store_dir_savename, accumulation_steps
	print(store_dir)
	print("Heloooooo")


	print("Distributed learning is", args.distributed)
	global best_acc1
	args.gpu = gpu

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		
		if args.multiprocessing_distributed:
			# For multiprocessing distributed training, rank needs to be the
			# global rank among all the processes
			args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size, rank=args.rank)
	

	# create model

	# if args.pretrained:
	#     print("=> using pre-trained model '{}'".format(args.arch))
	#     model = models.__dict__[args.arch](pretrained=True)
	# else:
	#     print("=> creating model '{}'".format(args.arch))
	#     model = models.__dict__[args.arch]()

	print('==> Building model..')
	with open(args.config_path, 'rb') as file:
		params = json.load(file)

	image_dim_size=224
	
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
	    net.fc = nn.Linear(num_ftrs, num_class)



	# print(model)
	total_params = count_parameters(model)
	# print("total parameters: ", total_params)

	if args.distributed:
		# For multiprocessing distributed, DistributedDataParallel constructor
		# should always set the single device scope, otherwise,
		# DistributedDataParallel will use all available devices.
		if args.gpu is not None:
			print("Setting distributed learning model to given gpu device, GPU ", args.gpu)
			print("Setting cuda device to: ", args.gpu)
			torch.cuda.set_device(args.gpu)
			model.cuda(args.gpu)
			# When using a single GPU per process and per
			# DistributedDataParallel, we need to divide the batch size
			# ourselves based on the total number of GPUs we have
			args.batch_size = int(args.batch_size / ngpus_per_node)
			args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		else:
			print("Setting distributed learning model to given gpu device, GPU ", args.gpu)
			model.cuda()
			# DistributedDataParallel will divide and allocate batch_size to all
			# available GPUs if device_ids are not set
			model = torch.nn.parallel.DistributedDataParallel(model)
	elif args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)
	else:
		print("No GPU number provided, adding data Parallel module")
		# DataParallel will divide and allocate batch_size to all available GPUs
		# if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
		#     model.features = torch.nn.DataParallel(model.features)
		#     model.cuda()
		# else:
		model = torch.nn.DataParallel(model).cuda()

	
	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda(args.gpu)
	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			if args.gpu is None:
				checkpoint = torch.load(args.resume)
			else:
				# Map model to be loaded to specified single gpu.
				loc = 'cuda:{}'.format(args.gpu)
				checkpoint = torch.load(args.resume, map_location=loc)
			args.start_epoch = checkpoint['epoch']
			best_acc1 = checkpoint['best_acc1']
			if args.gpu is not None:
				# best_acc1 may be from a checkpoint from a different GPU
				best_acc1 = best_acc1.to(args.gpu)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	# Data loading code
	traindir = os.path.join("/data/shargu/datasets/imagenet/ILSVRC/Data/CLS-LOC/", 'train')
	valdir = os.path.join("/data/shargu/datasets/imagenet/ILSVRC/Data/CLS-LOC", 'val')
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	train_dataset = datasets.ImageFolder(
		traindir,
		transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))

	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	else:
		train_sampler = None

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
		num_workers=args.workers, pin_memory=True, sampler=train_sampler)

	val_loader = torch.utils.data.DataLoader(
		datasets.ImageFolder(valdir, transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	if args.evaluate:
		validate(val_loader, model, criterion, args)
		return

	results = {'total_params': total_params,'args': args,'params': params,'train_acc': [],'test_acc': [], 'train_acc_5':[], 'test_acc_5':[]}
	
	for epoch in range(args.start_epoch, args.epochs):
		if args.distributed:
			train_sampler.set_epoch(epoch)
		adjust_learning_rate(optimizer, epoch, args)

		# train for one epoch
		epoch_train_acc,epoch_train_acc_5 =train(train_loader, model, criterion, optimizer, epoch, args)

		# evaluate on validation set
		acc1, acc5_val = validate(val_loader, model, criterion, args)
		results['train_acc'].append(epoch_train_acc)
		results['test_acc'].append(acc1)
		results['train_acc_5'].append(epoch_train_acc_5)
		results['test_acc_5'].append(acc5_val)
		# remember best acc@1 and save checkpoint
		is_best = acc1 > best_acc1
		best_acc1 = max(acc1, best_acc1)

		store_file = os.path.join(store_dir, 'performance.dct')
		pickle.dump(results, open(store_file, 'wb'))

		if not args.multiprocessing_distributed or (args.multiprocessing_distributed
				and args.rank % ngpus_per_node == 0):
			print(store_dir)
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'best_acc1': best_acc1,
				'optimizer' : optimizer.state_dict(),
			}, is_best=is_best, store_dir=store_dir)

	print("Finished Training")
	store_file = os.path.join(store_dir, 'performance.dct')
	pickle.dump(results, open(store_file, 'wb'))
	print("Saved performance results")

def train(train_loader, model, criterion, optimizer, epoch, args):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.4f')
	top5 = AverageMeter('Acc@5', ':6.4f')
	progress = ProgressMeter(
		len(train_loader),
		[batch_time, data_time, losses, top1, top5],
		prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.train()

	end = time.time()
	optimizer.zero_grad()
	for i, (images, target) in enumerate(train_loader):
		# if(i>26):
		# 	break
		# measure data loading time
		data_time.update(time.time() - end)

		if args.gpu is not None:
			images = images.cuda(args.gpu, non_blocking=True)
		target = target.cuda(args.gpu, non_blocking=True)

		# compute output
		output = model(images)
		# print(output.shape, target)
		loss = criterion(output, target)
		loss = loss / accumulation_steps

		# measure accuracy and record loss
		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), images.size(0))
		top1.update(acc1[0], images.size(0))
		top5.update(acc5[0], images.size(0))

		# compute gradient and do SGD step		
		loss.backward()

		if (i+1) % accumulation_steps == 0: 
			# print("Performed Gradient update")  
			optimizer.step()
			optimizer.zero_grad()
		

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			progress.display(i)

	return top1.avg, top5.avg



def train_withoutgradientaccumulation(train_loader, model, criterion, optimizer, epoch, args):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.4f')
	top5 = AverageMeter('Acc@5', ':6.4f')
	progress = ProgressMeter(
		len(train_loader),
		[batch_time, data_time, losses, top1, top5],
		prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.train()

	end = time.time()
	for i, (images, target) in enumerate(train_loader):
		# if(i>25):
		# 	break
		# measure data loading time
		data_time.update(time.time() - end)

		if args.gpu is not None:
			images = images.cuda(args.gpu, non_blocking=True)
		target = target.cuda(args.gpu, non_blocking=True)

		# compute output
		output = model(images)
		# print(output.shape, target)
		loss = criterion(output, target)

		# measure accuracy and record loss
		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), images.size(0))
		top1.update(acc1[0], images.size(0))
		top5.update(acc5[0], images.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()

		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			progress.display(i)

	return top1.avg, top5.avg

def validate(val_loader, model, criterion, args):
	batch_time = AverageMeter('Time', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.4f')
	top5 = AverageMeter('Acc@5', ':6.4f')
	progress = ProgressMeter(
		len(val_loader),
		[batch_time, losses, top1, top5],
		prefix='Test: ')

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (images, target) in enumerate(val_loader):
			# if(i>10):
			# 	break
			if args.gpu is not None:
				images = images.cuda(args.gpu, non_blocking=True)
			target = target.cuda(args.gpu, non_blocking=True)

			# compute output
			output = model(images)
			loss = criterion(output, target)

			# measure accuracy and record loss
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), images.size(0))
			top1.update(acc1[0], images.size(0))
			top5.update(acc5[0], images.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				progress.display(i)

		# TODO: this should also be done with the ProgressMeter
		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
			  .format(top1=top1, top5=top5))

	return top1.avg, top5.avg


def save_checkpoint(state, is_best, store_dir, filename='checkpoint.pth.tar'):
	torch.save(state, os.path.join(store_dir, filename))
	if is_best:
		print("Saving Best checkpoint")
		shutil.copyfile(os.path.join(store_dir, filename), os.path.join(store_dir, 'model_best.pth.tar'))


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

def count_parameters(model):
	# for name, param in model.named_parameters():
	#     if param.requires_grad:
	#         print(name, param.numel())
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
	main()