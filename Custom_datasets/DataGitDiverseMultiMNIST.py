import random
import torch
import numpy as np
from torchvision import datasets
from PIL import Image
from matplotlib import pyplot as plt

random.seed(4)
np.random.seed(4)
torch.manual_seed(4)
import torchvision.transforms as transforms

class DiverseMultiMNIST(datasets.MNIST):

    def __init__(self, root, train, transformation):
        super(DiverseMultiMNIST, self).__init__(root, train, download=True) 
        dataset = datasets.MNIST(root=root, train=train, transform = None, download=True)
        self.mnist_data = dataset.data
        self.mnist_labels = dataset.targets
        self.transform = transformation
        self.classes = list(range(10))

        print('Dataset Length is ', len(dataset.data))
        print("Classes: ", self.classes)

        # Generate label specific data  
        dict_indices_labels = {}
        for label in range(10):
            total_list = self.mnist_labels==torch.Tensor([label])
            res_list = list(filter(lambda x: total_list[x] == True, range(len(total_list)))) 
            dict_indices_labels[label] =  res_list
        self.dict_indices_labels = dict_indices_labels

    
    def __getitem__(self, index):
        
        
        img, label = self.mnist_data[index].numpy(), int(self.mnist_labels[index])
        img_pil = Image.fromarray(img, mode='L')

        # Convert to 36*36 By shifting and padding
        img = np.array(self.transform(img_pil)).astype(np.uint8)

        output_label = [0]*10
        train_transform = transforms.Compose([transforms.Grayscale(3),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                            ])


        decision = np.random.choice(['SingleDigit', 'OverlappingDigit'], 1, p=[0.167, 0.833])
        if decision[0] == 'SingleDigit':
            img = Image.fromarray(img, mode='L')
            output_label[label] = 1
            img  = train_transform(img) 
            output_label = torch.tensor(output_label, dtype=torch.float32)
            return img, output_label
        else:
            new_list = self.classes[0:label]+ self.classes[label+1:]
            digit_class = random.sample(new_list, 1)
            assert (digit_class[0] != label), "Overlapping images have same label"
            # data = self.mnist_data[self.mnist_labels==torch.Tensor(digit_class)]

            second_img = self.mnist_data[int(np.random.choice(self.dict_indices_labels[digit_class[0]], 1))].numpy()
            second_img_pil = Image.fromarray(second_img, mode='L')

            # Convert to 36*36 By shifting and padding
            second_img = np.array(self.transform(second_img_pil)).astype(np.uint8)
            
            # Overlay 
            merged = np.add(img, second_img, dtype=np.int32)
            merged = np.minimum(merged, 255).astype(np.uint8)
            
            # Output label
            output_label[digit_class[0]]=1
            output_label[label]=1
            # print(digit_class[0], label, output_label)
            # plt.imshow(merged, cmap='gray')
            # import os
            # os.chdir("/data/2015P002510/Sharut/MSR/ScalableCapsules/CIFAR100/Custom_datasets/")
            # plt.savefig("./check/"+str(index)+".png", dpi=200)
            # plt.close("all")

            # Final Transform 
            img = Image.fromarray(merged, mode='L')
            img  = train_transform(img)
            output_label = torch.tensor(output_label, dtype=torch.float32)
            return img, output_label
