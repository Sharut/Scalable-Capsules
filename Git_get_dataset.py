
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from torchvision.datasets import FashionMNIST
from torchvision.datasets import SVHN


from Custom_datasets.DataGitDiverseMultiMNIST import DiverseMultiMNIST
from Custom_datasets.DataMNIST import SimpleMNIST
from Custom_datasets.DataExpandedMNIST import ExpandedMNIST
from Custom_datasets.DataAffNIST import AffNIST
from utils import seed_torch
from torchvision import datasets
from Custom_datasets.DataAffNISTv2 import AffNISTv2

def get_dataset(name, seed, train_translation_rotation_list=None, test_translation_rotation_list=None):
	seed_torch(seed)

	if 'CIFAR' in name:
		print("Applying CIFAR transforms")
		transform_train = transforms.Compose([
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
		transform_test = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

		trainset = getattr(torchvision.datasets, name)(root='../data', train=True, download=True, transform=transform_train)
		testset = getattr(torchvision.datasets, name)(root='../data', train=False, download=True, transform=transform_test)
		num_class = int(name.split('CIFAR')[1])
		image_dim_size = 32
	
	elif name == 'MNIST':
		print("Applying Simple MNIST transforms")

		trainset = SimpleMNIST(pad=0, shift =2, root='../data', train=True)
		testset = SimpleMNIST(pad=0, shift =2, root='../data', train=False)
		num_class=10
		image_dim_size = 28

	elif name == 'AffineMNIST':
		print("Applying 28*28 Affine MNIST transforms")
		transform = transforms.Compose([transforms.RandomAffine(degrees=30, translate=(0.1,0.1)),transforms.Grayscale(3), transforms.ToTensor()])
		
		trainset = []
		testset = datasets.MNIST(root='../data/', train=False, download=True, transform=transform)
		num_class=10
		image_dim_size = 28

		



	elif name == 'DiverseMultiMNIST':
		DATAPATH = '../data/DiverseMultiMNIST/'
		print("Applying Diverse Multi MNIST transforms")
		for (translation,rotation) in train_translation_rotation_list:
			train_transform = transforms.Compose([
											transforms.Pad(4),
											transforms.RandomAffine(rotation,translation),
											])

			# trainset = torchvision.datasets.MNIST(DATAPATH, train=True, download=True, transform=train_transform)
			# print('len is ', (trainset.data[0].shape))
			trainset = DiverseMultiMNIST(root = DATAPATH+'train/', transformation = train_transform, train=True)
		
		for (translation,rotation) in test_translation_rotation_list:
			test_transform = transforms.Compose([
											transforms.Pad(4),
											transforms.RandomAffine(rotation,translation),
											])

			# trainset = torchvision.datasets.MNIST(DATAPATH, train=True, download=True, transform=train_transform)
			# print('len is ', (trainset.data[0].shape))
			testset = DiverseMultiMNIST(root = DATAPATH+'test/', transformation = test_transform, train=False)
			
			# testset = DiverseMultiMNIST(root = DATAPATH +'test/', transformation = test_transform, train=False)
		
		num_class=10
		image_dim_size = 36


	elif name == 'ExpandedMNIST':
		print("Applying Expanded MNIST transforms")
		trainset = ExpandedMNIST(root='../data', train=True, transformation = None)
		testset = ExpandedMNIST(root='../data', train=False, transformation = None)
		num_class=10
		image_dim_size = 40

	elif name == 'AffNIST':
		print("Applying Affine-MNIST transforms")
		transform_diverseMNIST = transforms.Compose([
				transforms.ToPILImage(),
				transforms.Pad(4), 
				transforms.ToTensor(),
			])
		# trainset = AffNIST(root='../data', train=True)
		trainset=[]
		testset = AffNIST(root='../data/AffNIST', train=False)
		num_class=10
		image_dim_size = 40

	elif name == 'FashionMNIST':
		print("Applying Fashion-MNIST transforms")
		"""Load fashionmnist dataset. The data is divided by 255 and subracted by mean and divided by standard deviation.
		"""
		DATAPATH = "../data/FashionMNIST/"
		train_loaders_desc = []
		test_loaders_desc = []
		for (translation,rotation) in train_translation_rotation_list:
			print("Augment train set with translation ", translation, " Rotation ", rotation)
			train_transform = transforms.Compose([
											transforms.RandomAffine(rotation,translation),
											transforms.Grayscale(3),
											transforms.ToTensor(),
											transforms.Normalize((0.5,), (0.5,)),
											])
			
			training_set = FashionMNIST(DATAPATH, train=True, download=True, transform=train_transform)
		
		for (translation,rotation) in test_translation_rotation_list:
			print("Augment test set with translation ", translation, " Rotation ", rotation)
			test_transform = transforms.Compose([
											transforms.RandomAffine(rotation,translation),
											transforms.Grayscale(3),
											transforms.ToTensor(),
											transforms.Normalize((0.5,), (0.5,)),
											])
			testing_set = FashionMNIST(DATAPATH, train=False, download=True, transform=test_transform)  
		num_class=10
		image_dim_size = 28
		return training_set, testing_set, num_class, image_dim_size





	elif name == 'Expanded_AffNISTv2':

		DATAPATH = "../data/AffineTest/"
		# train_translation_rotation_list = [((0.15,0.15))]#,((0.075,0.075),30),((0.075,0.075),60),((0.075,0.075),90),((0.075,0.075),180)]
		# test_translation_rotation_list = [((0,0),0)]
		for (translation,rotation) in train_translation_rotation_list:
			print("Augment train set with translation ", translation, " Rotation ", rotation)
			train_desc = 'train_'+str(translation[0])+'_'+str(translation[1])+'_'+str(rotation)+'_'+'MNIST'#for identifying in logs 
			train_transform = transforms.Compose([
											transforms.Pad(6),
											transforms.RandomAffine(rotation,translation),
											transforms.Grayscale(3),
											transforms.ToTensor(),
											transforms.Normalize((0.1307,), (0.3081,)),
											])
			trainset = torchvision.datasets.MNIST(DATAPATH+'MNIST', train=True, download=True, transform=train_transform)
		for (translation,rotation) in test_translation_rotation_list:
			test_desc = 'test_'+str(translation[0])+'_'+str(translation[1])+'_'+str(rotation)+'_'+'MNIST'#for identifying in logs  
			test_transform = transforms.Compose([
											#transforms.Resize(31),
											#transforms.RandomAffine(rotation,translation),
											transforms.Grayscale(3),
											transforms.ToTensor(),
											transforms.Normalize((0.1307,), (0.3081,)),
											])
			testset = AffNISTv2(DATAPATH+'AffNIST', train=False, transform=test_transform)  
		num_class=10
		image_dim_size = 40
		# testing_data_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True)

	elif name == 'SVHN':
		DATAPATH = "../data/SVHN/"
		train_loaders_desc = []
		test_loaders_desc = []
		for (translation,rotation) in train_translation_rotation_list:
			train_desc = 'train_'+str(translation[0])+'_'+str(translation[1])+'_'+str(rotation)+'_'+'SVHN'#for identifying in logs 
			train_transform = transforms.Compose([
											transforms.RandomAffine(rotation,translation,(0.9,1.1)),
											transforms.Grayscale(3),
											transforms.ToTensor(),
											transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
											])
			trainset = SVHN(DATAPATH, split='train', download=True, transform=train_transform)
		
		for (translation,rotation) in test_translation_rotation_list:
			test_desc = 'test_'+str(translation[0])+'_'+str(translation[1])+'_'+str(rotation)+'_'+'SVHN'#for identifying in logs  
			test_transform = transforms.Compose([
											#transforms.Resize(31),
											transforms.RandomAffine(rotation,translation),
											transforms.Grayscale(3),
											transforms.ToTensor(),
											transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
											])
			testset = SVHN(DATAPATH, split='test', download=True, transform=test_transform)  

		num_class = 10
		image_dim_size = 32
	return trainset, testset, num_class, image_dim_size
