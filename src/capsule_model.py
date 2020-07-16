#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
from src import layers, linformer
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import seed_torch
import warnings
# warnings.filterwarnings("ignore")
from torchprofile import profile_macs
'''
{'backbone': {'kernel_size': 3, 'output_dim': 128, 'input_dim': 3, 'stride': 2, 'padding': 1, 'out_img_size': 16}, 'primary_capsules': {'kernel_size': 1, 'stride': 1, 'input_dim': 128, 'caps_dim': 16, 'nu
m_caps': 32, 'padding': 0, 'out_img_size': 16}, 'capsules': [{'type': 'CONV', 'num_caps': 32, 'caps_dim': 16, 'kernel_size': 3, 'stride': 2, 'matrix_pose': True, 'out_img_size': 7}, {'type': 'CONV', 'num_
caps': 32, 'caps_dim': 16, 'kernel_size': 3, 'stride': 1, 'matrix_pose': True, 'out_img_size': 5}], 'class_capsules': {'num_caps': 10, 'caps_dim': 16, 'matrix_pose': True}}
{'kernel_size': 1, 'stride': 1, 'input_dim': 128, 'caps_dim': 16, 'num_caps': 32, 'padding': 0, 'out_img_size': 16}
'''





# Capsule model
class CapsModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True,
                 seed = 0):
        
        super(CapsModel, self).__init__()
        #### Parameters
        seed_torch(seed)
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'], seed=seed)
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100' or "NIST" in dataset or dataset == 'SVHN':
              print("Using standard ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'], seed=seed)
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'], seed=seed)
        
        print("Precaps: ", profile_macs(self.pre_caps, torch.randn(1,3,32,32)))
       
        ## Primary Capsule Layer (a single CNN)
        # {'kernel_size': 1, 'stride': 1, 'input_dim': 128, 'caps_dim': 16, 'num_caps': 32, 'padding': 0, 'out_img_size': 16}
        print(params['primary_capsules'])
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        print("PC Layer: ", profile_macs(self.pc_layer, torch.randn(1,128,16,16)))
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                output_img_size = params['capsules'][i]['out_img_size'] 

                # num_in_capsules=32, in_cap_d=16, out_Cap=32, out_dim_cap=16
                # 3x3 kernel, stride 2 and output shape: 7x7
                self.capsule_layers.append(
                    layers.CapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                seed=seed,
                                coordinate_add=False
                            )
                )
                x_input = torch.randn(1,in_n_caps,output_img_size,output_img_size,in_d_caps)
                print("Conv Capsule Layer ",i,' ', profile_macs(self.capsule_layers[i],x_input))
            

            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    # When there is no Conv layer after primary capsules
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']                                           
                
                elif params['capsules'][i-1]['type'] == 'CONV':
                    # There are a total of 14X14X32 capsule outputs, each being 16 dimensional 
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.CapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp,
                          seed=seed,
                          )
                )
                x_input = torch.randn(1,in_n_caps,1,in_d_caps)
                print("FC Capsule Layer ",i,' ', profile_macs(self.capsule_layers[i],x_input))                                                   

                                                               
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        
        self.capsule_layers.append(
            layers.CapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp,
                  seed=seed,
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        x_input = torch.randn(1,in_n_caps,1,in_d_caps)
        print("Class Capsule Layer ", profile_macs(self.capsule_layers[-1],x_input))
                
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)

        x_input = torch.randn(1,1,params['class_capsules']['num_caps'],params['class_capsules']['caps_dim'])
        print("Final FC Linear Layer ", profile_macs(self.final_fc,x_input))

        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))


    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        # Converts Input (b, 3, 14, 14)--> (b, 128, 14, 14)
        c = self.pre_caps(x)
        
        ## Primary Capsule Layer (a single CNN) (Ouput size: b, 512, 14, 14) (32 caps, 16 dim each)
        u = self.pc_layer(c)
        u = u.permute(0, 2, 3, 1) # b, 14, 14, 512
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # b, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # b, 32, 14, 14, 16
        
        # Layer norm
        init_capsule_value = self.nonlinear_act(u)  #capsule_utils.squash(u)

        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        

        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                
                # second to t iterations
                # perform the routing between the 2 capsule layers for some iterations 
                # till you move to next pair of layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        # Output capsule (last layer)
        out = capsule_values[-1]
        out = self.final_fc(out) # fixed classifier for all capsules
        # out = out.squeeze() # fixed classifier for all capsules
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules        
        return out 




# Capsule model with bilinear sparse routing
class CapsSAModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True,
                 seed = 0):
        
        super(CapsSAModel, self).__init__()
        #### Parameters
        seed_torch(seed)
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'], seed=seed)
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100'or "NIST" in dataset or dataset == 'SVHN':
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'], seed=seed)
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'], seed=seed)
        
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                self.capsule_layers.append(
                    layers.SACapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                seed=seed,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']                                           
                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.SACapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp,
                          seed=seed
                          )
                )                                                   
        
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            layers.SACapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp,
                  seed=seed
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print(c.shape)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
         
        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("helo ", out.shape)
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 


 # Capsule model with bilinear sparse routing and normal initialisation
class CapsRandomInitBAModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True,
                 seed = 0):
        
        super(CapsRandomInitBAModel, self).__init__()
        #### Parameters
        seed_torch(seed)
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'], seed=seed)
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100'or "NIST" in dataset or dataset == 'SVHN':
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'], seed=seed)
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'], seed=seed)
        
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                self.capsule_layers.append(
                    layers.RandomInitBACapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                seed=seed,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']                                           
                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.RandomInitBACapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp,
                          seed=seed
                          )
                )                                                   
        
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            layers.RandomInitBACapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp,
                  seed=seed
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print(c.shape)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
         
        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 





# Capsule model with bilinear routing without sinkhorn
class CapsBAModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True,
                 seed = 0):
        
        super(CapsBAModel, self).__init__()
        #### Parameters
        seed_torch(seed)
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'], seed=seed)
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100'or "NIST" in dataset or dataset == 'SVHN':
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'], seed=seed)
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'], seed=seed)
        
        print("Precaps: ", profile_macs(self.pre_caps, torch.randn(1,3,32,32)))
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        print("PC Layer: ", profile_macs(self.pc_layer, torch.randn(1,128,16,16)))
        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                output_img_size = params['capsules'][i]['out_img_size'] 
                self.capsule_layers.append(
                    layers.BACapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                seed=seed,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )
                x_input = torch.randn(1,in_n_caps,output_img_size,output_img_size,in_d_caps)
                print("Conv Capsule Layer ",i,' ', profile_macs(self.capsule_layers[i],x_input))
                            
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']                                           
                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.BACapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp,
                          seed=seed
                          )
                )                                                   
                x_input = torch.randn(1,in_n_caps,1,in_d_caps)
                print("FC Capsule Layer ",i,' ', profile_macs(self.capsule_layers[i],x_input))                                                   
                
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            layers.BACapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp,
                  seed=seed
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        x_input = torch.randn(1,in_n_caps,1,in_d_caps)
        print("Class Capsule Layer ", profile_macs(self.capsule_layers[-1],x_input))
                
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        
        x_input = torch.randn(1,1,params['class_capsules']['num_caps'],params['class_capsules']['caps_dim'])
        print("Final FC Linear Layer ", profile_macs(self.final_fc,x_input))

        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print(c.shape)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
         
        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 








# Capsule model with bilinear routing without sinkhorn with multi head attention
class CapsMultiHeadBAModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True,
                 seed = 0):
        
        super(CapsMultiHeadBAModel, self).__init__()
        #### Parameters
        seed_torch(seed)
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'], seed=seed)
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100'or "NIST" in dataset or dataset == 'SVHN':
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'], seed=seed)
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'], seed=seed)
        
        print("Precaps: ", profile_macs(self.pre_caps, torch.randn(1,3,32,32)))
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        print("PC Layer: ", profile_macs(self.pc_layer, torch.randn(1,128,16,16)))
        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['num_heads']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                output_img_size = params['capsules'][i]['out_img_size'] 
                self.capsule_layers.append(
                    layers.MultiHeadBACapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                num_heads = params['capsules'][i]['num_heads'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                seed=seed,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )
                x_input = torch.randn(1,in_n_caps,output_img_size,output_img_size,in_d_caps)
                print("Conv Capsule Layer ",i,' ', profile_macs(self.capsule_layers[i],x_input))
                            
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size'] 
                    in_d_caps = params['primary_capsules']['caps_dim']
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']* params['capsules'][i-1]['num_heads']
                    in_d_caps = params['capsules'][i-1]['caps_dim']                                           
                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['num_heads'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size'] 
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.MultiHeadBACapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          num_heads = params['capsules'][i]['num_heads'],
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp,
                          seed=seed
                          )
                )                                                   
                x_input = torch.randn(1,in_n_caps,1,in_d_caps)
                print("FC Capsule Layer ",i,' ', profile_macs(self.capsule_layers[i],x_input))                                                   
                
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['num_heads']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['num_heads'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            layers.MultiHeadBACapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  num_heads = params['class_capsules']['num_heads'],
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp,
                  seed=seed
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        x_input = torch.randn(1,in_n_caps,1,in_d_caps)
        print("Class Capsule Layer ", profile_macs(self.capsule_layers[-1],x_input))
                
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        
        x_input = torch.randn(1,1,params['class_capsules']['num_caps'],params['class_capsules']['caps_dim'])
        print("Final FC Linear Layer ", profile_macs(self.final_fc,x_input))

        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print(c.shape)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
         
        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 




# Capsule model with bilinear routing without sinkhorn (vector pose only)
class CapsBVAModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True,
                 seed = 0):
        
        super(CapsBVAModel, self).__init__()
        #### Parameters
        seed_torch(seed)
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'], seed=seed)
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100'or "NIST" in dataset:
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'], seed=seed)
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'], seed=seed)
        
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                self.capsule_layers.append(
                    layers.BVACapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                seed=seed,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']                                           
                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.BVACapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp,
                          seed=seed
                          )
                )                                                   
        
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            layers.BVACapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp,
                  seed=seed
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print(c.shape)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
         
        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 



# Capsule model with bilinear routing with dynamic routing
class CapsDBAModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True,
                 seed = 0):
        
        super(CapsDBAModel, self).__init__()
        #### Parameters
        seed_torch(seed)
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'],seed=seed)
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100'or "NIST" in dataset:
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'],seed=seed)
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'],seed=seed)
        
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                self.capsule_layers.append(
                    layers.DBACapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                seed=seed,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']                                           
                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.DBACapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp,
                          seed=seed
                          )
                )                                                   
        
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            layers.DBACapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp,
                  seed=seed
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print(c.shape)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        # init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
        init_capsule_value = u
         
        ## Main Capsule Layers 
        # Sequetial routing only
        if self.sequential_routing:
          
          capsule_values, _val = [init_capsule_value], init_capsule_value
          for i in range(len(self.capsule_layers)):
              routing_coeff = None
              for n in range(self.num_routing):
                  # print("Routing num ", n)
                  new_coeff, __val = self.capsule_layers[i].forward(_val, n, routing_coeff)
                  routing_coeff = new_coeff
              _val = __val
              capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 





# Capsule model with DYNAMIC ROUTING by Sara Sabour and Hinton
class CapsDRModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True,
                 seed = 0):
        
        super(CapsDRModel, self).__init__()
        #### Parameters
        seed_torch(seed)
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'],seed=seed)
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100'or "NIST" in dataset or dataset == 'SVHN':
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'],seed=seed)
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'],seed=seed)
        
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                self.capsule_layers.append(
                    layers.DRCapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                seed=seed,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']                                           
                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.DRCapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp,
                          seed=seed
                          )
                )                                                   
        
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            layers.DRCapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp,
                  seed=seed
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print(c.shape)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        # init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
        init_capsule_value = u
         
        ## Main Capsule Layers 
        # Sequetial routing only
        if self.sequential_routing:
          
          capsule_values, _val = [init_capsule_value], init_capsule_value
          for i in range(len(self.capsule_layers)):
              routing_coeff = None
              for n in range(self.num_routing):
                  # print("Routing num ", n)
                  new_coeff, __val = self.capsule_layers[i].forward(_val, n, routing_coeff)
                  routing_coeff = new_coeff
              _val = __val
              capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 















# MAIN IMPLEMENTATION FOR ARXIV PAPER

# Capsule model with bilinear routing and random projections
class CapsBilinearLocalLinformer(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 multi_transforms = False,
                 kernel_transformation = False,
                 sequential_routing=True,
                 seed = 0):
        
        super(CapsBilinearLocalLinformer, self).__init__()
        #### Parameters
        seed_torch(seed)
        self.sequential_routing = sequential_routing
        self.multi_transforms = multi_transforms
        self.kernel_transformation=kernel_transformation
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'],seed=seed)
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100' or 'NIST' in dataset or dataset == 'SVHN':
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'],seed=seed)
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'],seed=seed)
        
        
        
        print("Precaps: ", profile_macs(self.pre_caps, torch.randn(1,3,32,32)))
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     
        # print("PC Layer: ", profile_macs(self.pc_layer, torch.randn(1,128,16,16)))
        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                
                output_img_size = params['capsules'][i]['out_img_size']                    
                input_img_size = params['primary_capsules']['out_img_size'] if i==0 else \
                                                               params['capsules'][i-1]['out_img_size']




                self.capsule_layers.append(
                    layers.LACapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                input_img_size = input_img_size,
                                output_img_size = output_img_size,
                                multi_transforms = multi_transforms,
                                kernel_transformation = kernel_transformation,
                                hidden_dim= params['capsules'][i]['hidden_dim'],
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                seed=seed,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )

                x_input = torch.randn(1,in_n_caps,output_img_size,output_img_size,in_d_caps)
                print("Conv Capsule Layer ",i,' ', profile_macs(self.capsule_layers[i],x_input))
            
            elif params['capsules'][i]['type'] == 'FC':
                output_img_size = 1
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                    input_img_size = params['primary_capsules']['out_img_size']

                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                    input_img_size = 1

                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                    input_img_size = params['capsules'][i-1]['out_img_size']
                
                self.capsule_layers.append(
                    layers.LACapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          input_img_size = input_img_size,
                          output_img_size = output_img_size,
                          multi_transforms = multi_transforms,
                          kernel_transformation=kernel_transformation,
                          hidden_dim= params['capsules'][i]['hidden_dim'],
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp,
                          seed=seed
                          )
                )

                x_input = torch.randn(1,in_n_caps,1,in_d_caps)
                print("FC Capsule Layer ",i,' ', profile_macs(self.capsule_layers[i],x_input))                                                   
        
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            output_img_size = 1
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
                input_img_size = 1
                
            
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
                input_img_size = params['capsules'][-1]['out_img_size']

            

                
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
            input_img_size = params['primary_capsules']['out_img_size']

        
        
        self.capsule_layers.append(
            layers.LACapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  input_img_size = input_img_size,
                  output_img_size = output_img_size,
                  multi_transforms = multi_transforms, 
                  kernel_transformation=kernel_transformation,
                  hidden_dim= params['class_capsules']['hidden_dim'],
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp,
                  seed=seed
                  )
        )

        x_input = torch.randn(1,in_n_caps,1,in_d_caps)
        print("Class Capsule Layer ", profile_macs(self.capsule_layers[-1],x_input))
        
        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)

        x_input = torch.randn(1,1,params['class_capsules']['num_caps'],params['class_capsules']['caps_dim'])
        print("Final FC Linear Layer ", profile_macs(self.final_fc,x_input))

        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])

        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
         
        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # print(i, ' ', _val.shape)
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 






# Capsule model with bilinear routing and linformer with multiple HEADS per capsule type
class CapsMultiHeadBilinearLocalLinformer(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 multi_transforms = False,
                 kernel_transformation=False,
                 sequential_routing=True,
                 seed = 0):
        
        super(CapsMultiHeadBilinearLocalLinformer, self).__init__()
        #### Parameters
        seed_torch(seed)
        self.sequential_routing = sequential_routing
        self.kernel_transformation=kernel_transformation
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'],seed=seed)
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100' or 'NIST' in dataset or dataset == 'SVHN':
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'],seed=seed)
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'],seed=seed)
        
        
        
        print("Precaps: ", profile_macs(self.pre_caps, torch.randn(1,3,32,32)))
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     
        print("PC Layer: ", profile_macs(self.pc_layer, torch.randn(1,128,16,16)))
        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               (params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['num_heads'] )
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                
                output_img_size = params['capsules'][i]['out_img_size']                    
                input_img_size = params['primary_capsules']['out_img_size'] if i==0 else \
                                                               params['capsules'][i-1]['out_img_size']




                self.capsule_layers.append(
                    layers.MultiHeadLACapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                input_img_size = input_img_size,
                                output_img_size = output_img_size,
                                num_heads= params['capsules'][i]['num_heads'],
                                multi_transforms=multi_transforms,
                                kernel_transformation = kernel_transformation,
                                hidden_dim= params['capsules'][i]['hidden_dim'],
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                seed=seed,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )

                x_input = torch.randn(1,in_n_caps,output_img_size,output_img_size,in_d_caps)
                print("Conv Capsule Layer ",i,' ', profile_macs(self.capsule_layers[i],x_input))
            
            elif params['capsules'][i]['type'] == 'FC':
                output_img_size = 1
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                    input_img_size = params['primary_capsules']['out_img_size']

                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['num_heads'] 
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                    input_img_size = 1

                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['num_heads'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                    input_img_size = params['capsules'][i-1]['out_img_size']
                
                self.capsule_layers.append(
                    layers.MultiHeadLACapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          input_img_size = input_img_size,
                          output_img_size = output_img_size,
                          multi_transforms=multi_transforms,
                          kernel_transformation = kernel_transformation,
                          num_heads= params['capsules'][i]['num_heads'],
                          hidden_dim= params['capsules'][i]['hidden_dim'],
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp,
                          seed=seed
                          )
                )

                x_input = torch.randn(1,in_n_caps,1,in_d_caps)
                print("FC Capsule Layer ",i,' ', profile_macs(self.capsule_layers[i],x_input))                                                   
        
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            output_img_size = 1
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['num_heads'] 
                in_d_caps = params['capsules'][-1]['caps_dim']
                input_img_size = 1
                
            
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['num_heads'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
                input_img_size = params['capsules'][-1]['out_img_size']

            

                
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
            input_img_size = params['primary_capsules']['out_img_size']

        
        
        self.capsule_layers.append(
            layers.MultiHeadLACapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  input_img_size = input_img_size,
                  output_img_size = output_img_size,
                  num_heads= params['class_capsules']['num_heads'],
                  multi_transforms=multi_transforms,
                  kernel_transformation = kernel_transformation,
                  hidden_dim= params['class_capsules']['hidden_dim'],
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp,
                  seed=seed
                  )
        )

        x_input = torch.randn(1,in_n_caps,1,in_d_caps)
        print("Class Capsule Layer ", profile_macs(self.capsule_layers[-1],x_input))
        
        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)

        x_input = torch.randn(1,1,params['class_capsules']['num_caps'],params['class_capsules']['caps_dim'])
        print("Final FC Linear Layer ", profile_macs(self.final_fc,x_input))

        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])

        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
         
        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # print(i, ' ', _val.shape)
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 



# Global Linformer (only FC, No Convolutions)
# Capsule model
class CapsBilinearGlobalLinformerModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True,
                 seed = 0):
        
        super(CapsBilinearGlobalLinformerModel, self).__init__()
        #### Parameters
        seed_torch(seed)

        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'],seed=seed)
        
        elif backbone == 'resnet':
            if dataset == 'CIFAR10' or dataset == 'CIFAR100':
              print("Using standard ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'],seed=seed)
            else:
              print("Using ImageNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'],seed=seed)
        
        
        ## Primary Capsule Layer (a single CNN)
        # {'kernel_size': 1, 'stride': 1, 'input_dim': 128, 'caps_dim': 16, 'num_caps': 32, 'padding': 0, 'out_img_size': 16}
        print(params['primary_capsules'])
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                # num_in_capsules=32, in_cap_d=16, out_Cap=32, out_dim_cap=16
                # 3x3 kernel, stride 2 and output shape: 7x7
                self.capsule_layers.append(
                    layers.CapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                seed=seed,
                                coordinate_add=False
                            )
                )
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    # When there is no Conv layer after primary capsules
                    # in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                    #                                                                         params['primary_capsules']['out_img_size']
                    
                    in_n_caps = params['primary_capsules']['num_caps']
                    in_d_caps = params['primary_capsules']['caps_dim']
                    input_size = params['primary_capsules']['out_img_size']
                
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim'] 
                    input_size = params['capsules'][i-1]['h_out']                                           
                
                elif params['capsules'][i-1]['type'] == 'CONV':
                    # There are a total of 14X14X32 capsule outputs, each being 16 dimensional 
                    # in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                    #                                                                        params['capsules'][i-1]['out_img_size']  
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                    input_size = params['capsules'][i-1]['out_img_size']
                self.capsule_layers.append(
                    linformer.BilinearGlobalLinformerDebugCapsuleFC(hidden_dim= params['capsules'][i]['hidden_dim'],
                          input_size = input_size,
                          in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          h_out = params['capsules'][i]['h_out'],
                          child_kernel_size=params['capsules'][i]['child_kernel_size'],
                          child_stride=params['capsules'][i]['child_stride'],
                          child_padding=params['capsules'][i]['child_padding'],
                          parent_kernel_size=params['capsules'][i]['parent_kernel_size'],
                          parent_stride=params['capsules'][i]['parent_stride'],
                          parent_padding=params['capsules'][i]['parent_padding'],
                          parameter_sharing = params['capsules'][i]['parameter_sharing'],
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp,
                          seed=seed
                          )
                )
                                                               
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
                input_size=params['capsules'][-1]['h_out']

            elif params['capsules'][-1]['type'] == 'CONV':    
                # in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                #                                                                    params['capsules'][-1]['out_img_size']
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
                input_size = params['capsules'][i-1]['out_img_size']
        else:
            # in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
            #                                                                    params['primary_capsules']['out_img_size']
            in_n_caps = params['primary_capsules']['num_caps']
            in_d_caps = params['primary_capsules']['caps_dim']
            input_size = params['primary_capsules']['out_img_size']
        
        self.capsule_layers.append(
            linformer.BilinearGlobalLinformerDebugCapsuleFC(hidden_dim= params['class_capsules']['hidden_dim'],
                    input_size = input_size,
                    in_n_capsules=in_n_caps, 
                    in_d_capsules=in_d_caps, 
                    out_n_capsules=params['class_capsules']['num_caps'], 
                    out_d_capsules=params['class_capsules']['caps_dim'], 
                    h_out = params['class_capsules']['h_out'],
                    child_kernel_size=params['capsules'][i]['child_kernel_size'],
                    child_stride=params['capsules'][i]['child_stride'],
                    child_padding=params['capsules'][i]['child_padding'],
                    parent_kernel_size=params['capsules'][i]['parent_kernel_size'],
                    parent_stride=params['capsules'][i]['parent_stride'],
                    parent_padding=params['capsules'][i]['parent_padding'],
                    parameter_sharing = params['class_capsules']['parameter_sharing'],
                    matrix_pose=params['class_capsules']['matrix_pose'],
                    dp=dp,
                    seed=seed
                    )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))


    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        # Converts Input (b, 3, 14, 14)--> (b, 128, 14, 14)
        c = self.pre_caps(x)
        
        ## Primary Capsule Layer (a single CNN) (Ouput size: b, 512, 14, 14) (32 caps, 16 dim each)
        u = self.pc_layer(c)
        u = u.permute(0, 2, 3, 1) # b, 14, 14, 512
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # b, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # b, 32, 14, 14, 16
        
        # Layer norm
        init_capsule_value = self.nonlinear_act(u)  #capsule_utils.squash(u)

        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
              # print("I is", i)
              _val = self.capsule_layers[i].forward(_val, 0)
              capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                  _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                  capsule_values[i+1])
                  _capsule_values.append(_val)
                capsule_values = _capsule_values
        

        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                
                # second to t iterations
                # perform the routing between the 2 capsule layers for some iterations 
                # till you move to next pair of layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        # Output capsule (last layer)
        out = capsule_values[-1]
        out = self.final_fc(out) # fixed classifier for all capsules
        out = out.squeeze() # fixed classifier for all capsules
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules        
        return out 

