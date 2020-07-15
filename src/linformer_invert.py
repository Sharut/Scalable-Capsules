import math
import torch
from torch import nn
from operator import mul
from fractions import gcd
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, wraps, reduce
import numpy as np

########## Linformer projection on each kernel
class LocalLinformerProjection(nn.Module):
	def __init__(self, 
				in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
				matrix_pose, layer_type, input_img_size, output_img_size, hidden_dim=None, kernel_size=None, kernel_transformation = False, parameter_sharing='headwise',
				dropout = 0.):
		super().__init__()
  
		self.in_d_capsules = in_d_capsules
		self.out_d_capsules = out_d_capsules
		self.in_n_capsules = in_n_capsules
		self.out_n_capsules = out_n_capsules
		self.input_img_size=input_img_size
		self.output_img_size=output_img_size
		self.hidden_dim=hidden_dim
		self.pose_dim = in_d_capsules
		self.layer_type = layer_type
		self.kernel_size = kernel_size
		self.matrix_pose = matrix_pose
		self.parameter_sharing = parameter_sharing
		self.kernel_transformation = kernel_transformation

		if self.layer_type == 'FC':
			self.kernel_size=1

		if matrix_pose:
			# Random Initialisation of Two matrices
			self.matrix_pose_dim = int(np.sqrt(self.in_d_capsules))
			
			# w_current =(3,3,32,4,4)
			self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
													 in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
			self.w_next = nn.Parameter(0.02*torch.randn(
													 out_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
		else:
			self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
													 in_n_capsules, self.pose_dim, self.pose_dim))
			self.w_next = nn.Parameter(0.02*torch.randn(
													 out_n_capsules, self.pose_dim, self.pose_dim))

		
		max_seq_len = self.kernel_size*self.kernel_size*self.in_n_capsules
		heads = 1
		
		if parameter_sharing == "headwise":
			# print("Hello")
			self.E_proj = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
													 in_n_capsules, hidden_dim))
	
		else:
			assert (False),"Yet to write the non-headwise method"
		

		# Positional embeddings: (7,7,16)
		# self.rel_embedd = None
		self.dropout = nn.Dropout(dropout)
		print("You are using Bilinear routing with Linformer")


	def forward(self, current_pose, h_out=1, w_out=1, next_pose=None):
		
		# print('Using linformer kernels')
		# current pose: (b,32,3,3,7,7,16)
		# if FC current pose is (b, numcaps*h_in*w_in, caps_dim)
		if next_pose is None:
			# ist iteration
			batch_size = current_pose.shape[0]
			if self.layer_type=='conv':
				# (b, h_out, w_out, num_capsules, kernel_size, kernel_size, capsule_dim)
				# (b,7,7,32,3,3,16)
				current_pose = current_pose.permute([0,4,5,1,2,3,6])
				h_out = h_out
				w_out = w_out
			
			elif self.layer_type=='FC':
				h_out = 1
				w_out = 1

			pose_dim = self.pose_dim
			w_current = self.w_current
			w_next = self.w_next
			if self.matrix_pose:
				#w_current =(3,3,32,4,4) --> (3*3*32, 4, 4)
				w_current = w_current.reshape(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				w_current = w_current.reshape(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim)
			
			#
			# W_current is C_{L} and w_next is N_{L}
			w_current = w_current.unsqueeze(0)  
			w_next = w_next.unsqueeze(0)

			current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)#view error
			
			if self.matrix_pose:
				# (b*7*7, 3*3*32, 4, 4) = (49b, 288, 4, 4)
				# print(current_pose.shape)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
			else:
				current_pose = current_pose.unsqueeze(2)
			
			# Multiplying p{L} by C_{L} to change to c_{L}
			# Current pose: (49b, 288, 4, 4), w_current = (1, 288, 4, 4)
			# Same matrix for the entire batch, output  = (49b, 288, 4, 4)
			current_pose = torch.matmul(current_pose, w_current) 


			if self.matrix_pose:
				# Current_pose = (49b, 288, 16)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
			else:
				current_pose = current_pose.squeeze(2)
			
			############## Don't apply linformer for 1st iteration
			#current_pose = current_pose.permute(2,0,1) # (16,49b,288)
			#E_proj = self.E_proj.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.hidden_dim) # (288, hidden_dim)            

			#current_pose = torch.matmul(current_pose, E_proj) # (16,49b,hidden_dim)
			#current_pose = current_pose.permute(1,2,0) # (49b, hidden_Dim, 16)

	
			# R_{i,j} = (49b, m, 288)
			#dots=(torch.ones(batch_size*h_out*w_out, self.out_n_capsules, self.hidden_dim)* (pose_dim ** -0.5)).type_as(current_pose).to(current_pose)
			#dots = dots.softmax(dim=-2)
			#value = np.unique(dots.cpu().numpy())[0]

			value = (float)(1.0/self.out_n_capsules) 
 
			next_pose_candidates = current_pose  
			#next_pose_candidates = torch.einsum('bij,bje->bie', dots, next_pose_candidates)
			
			######### Complexity reduction step instead of einsum
			next_pose_candidates=torch.sum(next_pose_candidates, dim=1) * value
			next_pose_candidates = next_pose_candidates.unsqueeze(1)
			next_pose_candidates = next_pose_candidates.expand(next_pose_candidates.shape[0], self.out_n_capsules, next_pose_candidates.shape[2])

			#
			next_pose_candidates = next_pose_candidates.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
			next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4)
			next_pose_candidates = next_pose_candidates.reshape(-1,next_pose_candidates.shape[3], next_pose_candidates.shape[4])
			

			if self.matrix_pose:
				# Correct shapes: (49b, m, 4, 4)
				next_pose_candidates = next_pose_candidates.reshape(next_pose_candidates.shape[0], next_pose_candidates.shape[1], self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose_candidates = next_pose_candidates.unsqueeze(2)
			
			# Found final pose of next layer by multiplying X with N_{L}
			# Multiply (49b, m, 4, 4) with (1, m, 4, 4) == (49b, m , 4, 4)
			next_pose_candidates = torch.matmul(next_pose_candidates, w_next)

			# Reshape: (b, 7, 7, m, 16)
			next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
			
		

			if self.layer_type == 'conv':
				# Reshape: (b,m,7,7,16) (just like original input, without expansion)
				next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
			
			elif self.layer_type == 'FC':
				# Reshape: (b, 1, 1, m, 16) --> (b, 1, m, 16) (h_out, w_out ==1)
				next_pose_candidates = next_pose_candidates.squeeze(1)
			return next_pose_candidates
		

		else:
			# 2nd to T iterations
			batch_size = next_pose.shape[0]
			if self.layer_type=='conv':
				# Current_pose = (b,7,7,32,3,3,16)
				current_pose = current_pose.permute([0,4,5,1,2,3,6])
				
				# next_pose = (b,m,7,7,16) --> (b,7,7,m,16)
				next_pose = next_pose.permute([0,2,3,1,4])
				h_out = next_pose.shape[1]
				w_out = next_pose.shape[2]
		   
			elif self.layer_type=='FC':
				h_out = 1
				w_out = 1
			
			pose_dim = self.pose_dim
			w_current = self.w_current
			w_next = self.w_next
			if self.matrix_pose:
				# w_current = (288,4,4)
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim) 
			
			# w_current = (1,288,4,4)
			w_current = w_current.unsqueeze(0)  
			w_next = w_next.unsqueeze(0)
			
			
			current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)            
			if self.matrix_pose:
				# Current_pose = (49b, 288, 4, 4)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
			else:
				current_pose = current_pose.unsqueeze(2)
			
			# Tranformed currentlayer capsules to c_{L}
			# Multiply (49b, 288, 4, 4) with (1,288,4,4) --> (49b, 288, 4, 4)
			current_pose = torch.matmul(current_pose, w_current)
			
			if self.matrix_pose:
				# Current_pose = (49b, 288, 16)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
			else:
				current_pose = current_pose.squeeze(2)

			
			############## Linformer Projection
			if self.kernel_transformation == False:
				current_pose = current_pose.permute(2,0,1) # (16,49b,288)
				E_proj = self.E_proj.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.hidden_dim) # (288, hidden_dim)            

				current_pose = torch.matmul(current_pose, E_proj) # (16,49b,hidden_dim)
				current_pose = current_pose.permute(1,2,0) # (49b, hidden_Dim, 16)
				# print(current_pose.shape, self.in_n_capsules, self.hidden_dim)
					
			else:
				if self.layer_type == 'conv':
					current_pose = current_pose.reshape(self.pose_dim, batch_size*h_out*w_out, self.in_n_capsules, self.kernel_size*self.kernel_size).unsqueeze(3) # (16,49b,32,1,9)
					E_proj = self.E_proj.view(self.in_n_capsules, self.kernel_size*self.kernel_size, self.hidden_dim) # (32,9,hidden_dim)            
					current_pose = torch.matmul(current_pose, E_proj) # (16,49b,32,1,hidden_dim)
					current_pose = current_pose.view(current_pose.shape[0], current_pose.shape[1], -1) # (16,49b, 32*hidden_dim)
					current_pose = current_pose.permute(1,2,0) # (49b, hidden_Dim, 16)
					
				else:
					
					current_pose = current_pose.permute(2,0,1) # (16,b,800)
					E_proj = self.E_proj.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.hidden_dim) # (800, hidden_dim)            
					current_pose = torch.matmul(current_pose, E_proj) # (16,b,hidden_dim)
					current_pose = current_pose.permute(1,2,0) # (b, hidden_Dim, 16)
					# print(current_pose.shape, self.in_n_capsules, self.hidden_dim)


								

			###################### Positonal Embeddings
			# Adding positional embeddings to next pose: (b,7,7,m,16) -->(b,m,7,7,16)+(7,7,16)
			# print("original ", next_pose.shape)
			next_pose = next_pose.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
			# print(next_pose.shape, self.rel_embedd.shape)
			# next_pose = next_pose + self.rel_embedd
				

			# next_pose = (b,m,7,7,16) --> (49b,m,16)   
			next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
			
			if self.matrix_pose:
				# next_pose = (49b,m,16)  -->  (49b,m,4,4) 
				next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose = next_pose.unsqueeze(3)
			
			# Tranform next pose using N_{L}: w_next = (49b,m,4,4) * (1,m,4,4)
			next_pose = torch.matmul(w_next, next_pose)
			

			if self.matrix_pose:
				# next_pose = (49b,m,16)
				next_pose = next_pose.view(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
			else:
				next_pose = next_pose.squeeze(3)
	
			# Finding scaled alignment scores between updated buckets
			# dots = (49b, m ,288)
			dots = torch.einsum('bje,bie->bji', next_pose, current_pose) * (pose_dim ** -0.5) 
			

			# attention routing along dim=-2 (next layer buckets)
			# Dim=-1 if you wanna invert the inverted attention
			# print("INVERTEDDDD")
			dots = dots.softmax(dim=-1) 
			next_pose_candidates = current_pose

			# Yet to multiply with N_{L} (next_w)
			next_pose_candidates = torch.einsum('bji,bie->bje', dots, next_pose_candidates)
			
			if self.matrix_pose:
				# next pose: 49b,m,16 --> 49b,m,4,4
				next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1],self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose_candidates = next_pose_candidates.unsqueeze(3)
			
			# Multiplied with N_{j} to get final pose
			# w_next: (49b,m,4,4); b_next_pose_candidates: (49b,m , 4, 4)
			next_pose_candidates = torch.matmul(next_pose_candidates, w_next)
			
			# next_pose_candidates = (b,7,7,m,16)
			next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
			

			if self.layer_type == 'conv':
				# next_pose_candidates = (b,m,7,7,16)
				next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
			elif self.layer_type == 'FC':
				# next_pose_candidates = (b,1,1,m,16) --> (b,1,m,16)
				next_pose_candidates = next_pose_candidates.squeeze(1)
			return next_pose_candidates







########## Linformer projection on each kernel
class MithunSirLocalLinformerProjection(nn.Module):
	def __init__(self, 
				in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
				matrix_pose, layer_type, input_img_size, output_img_size, hidden_dim=None, kernel_size=None, parameter_sharing='headwise',
				dropout = 0.):
		super().__init__()
  
		self.in_d_capsules = in_d_capsules
		self.out_d_capsules = out_d_capsules
		self.in_n_capsules = in_n_capsules
		self.out_n_capsules = out_n_capsules
		self.input_img_size=input_img_size
		self.output_img_size=output_img_size
		self.hidden_dim=hidden_dim
		self.pose_dim = in_d_capsules
		self.layer_type = layer_type
		self.kernel_size = kernel_size
		self.matrix_pose = matrix_pose
		self.parameter_sharing = parameter_sharing

		if self.layer_type == 'FC':
			self.kernel_size=1

		if matrix_pose:
			# Random Initialisation of Two matrices
			self.matrix_pose_dim = int(np.sqrt(self.in_d_capsules))
			
			# w_current =(3,3,32,4,4)
			self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
													 in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
			self.w_next = nn.Parameter(0.02*torch.randn(
													 out_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
		else:
			self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
													 in_n_capsules, self.pose_dim, self.pose_dim))
			self.w_next = nn.Parameter(0.02*torch.randn(
													 out_n_capsules, self.pose_dim, self.pose_dim))

		
		max_seq_len = self.kernel_size*self.kernel_size*self.in_n_capsules
		heads = 1
		
		if parameter_sharing == "headwise":
			# print("Hello")
			self.E_proj = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
													 in_n_capsules, hidden_dim))
	
		else:
			assert (False),"Yet to write the non-headwise method"
		

		# Positional embeddings: (7,7,16)
		# self.rel_embedd = None
		self.dropout = nn.Dropout(dropout)
		print("You are using Bilinear routing with Linformer")


	def forward(self, current_pose, h_out=1, w_out=1, next_pose=None):
		
		# print('Using linformer kernels')
		# current pose: (b,32,3,3,7,7,16)
		# if FC current pose is (b, numcaps*h_in*w_in, caps_dim)
		if next_pose is None:
			# ist iteration
			batch_size = current_pose.shape[0]
			if self.layer_type=='conv':
				# (b, h_out, w_out, num_capsules, kernel_size, kernel_size, capsule_dim)
				# (b,7,7,32,3,3,16)
				current_pose = current_pose.permute([0,4,5,1,2,3,6])
				h_out = h_out
				w_out = w_out
			
			elif self.layer_type=='FC':
				h_out = 1
				w_out = 1

			pose_dim = self.pose_dim
			w_current = self.w_current
			w_next = self.w_next
			if self.matrix_pose:
				#w_current =(3,3,32,4,4) --> (3*3*32, 4, 4)
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim)
			
			#
			# W_current is C_{L} and w_next is N_{L}
			w_current = w_current.unsqueeze(0)  
			w_next = w_next.unsqueeze(0)

			current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)#view error
			
			if self.matrix_pose:
				# (b*7*7, 3*3*32, 4, 4) = (49b, 288, 4, 4)
				# print(current_pose.shape)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
			else:
				current_pose = current_pose.unsqueeze(2)
			
			################ NO MATRIX MULTIPLICATION ON CHILD CAPSULES
			# current_pose = torch.matmul(current_pose, w_current) 


			if self.matrix_pose:
				# Current_pose = (49b, 288, 16)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
			else:
				current_pose = current_pose.squeeze(2)
			
			############## Linformer Projection
			current_pose = current_pose.permute(2,0,1) # (16,49b,288)
			E_proj = self.E_proj.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.hidden_dim) # (288, hidden_dim)            

			current_pose = torch.matmul(current_pose, E_proj) # (16,49b,hidden_dim)
			current_pose = current_pose.permute(1,2,0) # (49b, hidden_Dim, 16)

	
			# R_{i,j} = (49b, m, 288)
			dots=(torch.ones(batch_size*h_out*w_out, self.out_n_capsules, self.hidden_dim)* (pose_dim ** -0.5)).type_as(current_pose).to(current_pose)
			dots = dots.softmax(dim=-2)
			
 
			next_pose_candidates = current_pose  
			# Multiplies r_{i,j} with c_{L} ( no sorting in the 1st iteration) to give X. Still have to
			# multiply with N_{L} 
			# next pose: (49b, m, 16) 
			next_pose_candidates = torch.einsum('bij,bje->bie', dots, next_pose_candidates)
			
			###################### Positional Embeddings
			# (49b,m,16) --> (b,m,7,7,16) + rel_embedding (7,7,16) and then reshaped to (49b,m,16)
			next_pose_candidates = next_pose_candidates.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
			# next_pose_candidates = next_pose_candidates + self.rel_embedd
			next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4)
			next_pose_candidates = next_pose_candidates.reshape(-1,next_pose_candidates.shape[3], next_pose_candidates.shape[4])
			
			if self.matrix_pose:
				# Correct shapes: (49b, m, 4, 4)
				next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1], self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose_candidates = next_pose_candidates.unsqueeze(2)
			
			# Found final pose of next layer by multiplying X with N_{L}
			# Multiply (49b, m, 4, 4) with (1, m, 4, 4) == (49b, m , 4, 4)
			next_pose_candidates = torch.matmul(next_pose_candidates, w_next)

			# Reshape: (b, 7, 7, m, 16)
			next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
			
		

			if self.layer_type == 'conv':
				# Reshape: (b,m,7,7,16) (just like original input, without expansion)
				next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
			
			elif self.layer_type == 'FC':
				# Reshape: (b, 1, 1, m, 16) --> (b, 1, m, 16) (h_out, w_out ==1)
				next_pose_candidates = next_pose_candidates.squeeze(1)
			return next_pose_candidates
		

		else:
			# 2nd to T iterations
			batch_size = next_pose.shape[0]
			if self.layer_type=='conv':
				# Current_pose = (b,7,7,32,3,3,16)
				current_pose = current_pose.permute([0,4,5,1,2,3,6])
				
				# next_pose = (b,m,7,7,16) --> (b,7,7,m,16)
				next_pose = next_pose.permute([0,2,3,1,4])
				h_out = next_pose.shape[1]
				w_out = next_pose.shape[2]
		   
			elif self.layer_type=='FC':
				h_out = 1
				w_out = 1
			
			pose_dim = self.pose_dim
			w_current = self.w_current
			w_next = self.w_next
			if self.matrix_pose:
				# w_current = (288,4,4)
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim) 
			
			# w_current = (1,288,4,4)
			w_current = w_current.unsqueeze(0)  
			w_next = w_next.unsqueeze(0)
			
			
			current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)            
			if self.matrix_pose:
				# Current_pose = (49b, 288, 4, 4)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
			else:
				current_pose = current_pose.unsqueeze(2)
			
			# Tranformed currentlayer capsules to c_{L}
			############## NO TRANSFORMATION FOR CHILD CAPSULES
			# current_pose = torch.matmul(current_pose, w_current)
			
			if self.matrix_pose:
				# Current_pose = (49b, 288, 16)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
			else:
				current_pose = current_pose.squeeze(2)

			
			############## Linformer Projection
			current_pose = current_pose.permute(2,0,1) # (16,49b,288)
			E_proj = self.E_proj.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.hidden_dim) # (288, hidden_dim)            

			current_pose = torch.matmul(current_pose, E_proj) # (16,49b,hidden_dim)
			current_pose = current_pose.permute(1,2,0) # (49b, hidden_Dim, 16)


			###################### Positonal Embeddings
			# Adding positional embeddings to next pose: (b,7,7,m,16) -->(b,m,7,7,16)+(7,7,16)
			# print("original ", next_pose.shape)
			next_pose = next_pose.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
			# print(next_pose.shape, self.rel_embedd.shape)
			# next_pose = next_pose + self.rel_embedd
				

			# next_pose = (b,m,7,7,16) --> (49b,m,16)   
			next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
			
			if self.matrix_pose:
				# next_pose = (49b,m,16)  -->  (49b,m,4,4) 
				next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose = next_pose.unsqueeze(3)
			
			# Tranform next pose using N_{L}: w_next = (49b,m,4,4) * (1,m,4,4)
			next_pose = torch.matmul(w_next, next_pose)
			

			if self.matrix_pose:
				# next_pose = (49b,m,16)
				next_pose = next_pose.view(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
			else:
				next_pose = next_pose.squeeze(3)
	
			# Finding scaled alignment scores between updated buckets
			# dots = (49b, m ,288)
			dots = torch.einsum('bje,bie->bji', next_pose, current_pose) * (pose_dim ** -0.5) 
			

			# attention routing along dim=-2 (next layer buckets)
			# Dim=-1 if you wanna invert the inverted attention
			dots = dots.softmax(dim=-2) 
			next_pose_candidates = current_pose

			# Yet to multiply with N_{L} (next_w)
			next_pose_candidates = torch.einsum('bji,bie->bje', dots, next_pose_candidates)
			
			if self.matrix_pose:
				# next pose: 49b,m,16 --> 49b,m,4,4
				next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1],self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose_candidates = next_pose_candidates.unsqueeze(3)
			
			# Multiplied with N_{j} to get final pose
			# w_next: (49b,m,4,4); b_next_pose_candidates: (49b,m , 4, 4)
			next_pose_candidates = torch.matmul(next_pose_candidates, w_next)
			
			# next_pose_candidates = (b,7,7,m,16)
			next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
			

			if self.layer_type == 'conv':
				# next_pose_candidates = (b,m,7,7,16)
				next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
			elif self.layer_type == 'FC':
				# next_pose_candidates = (b,1,1,m,16) --> (b,1,m,16)
				next_pose_candidates = next_pose_candidates.squeeze(1)
			return next_pose_candidates






class BilinearProjectionWithEmbeddings(nn.Module):
	def __init__(self, 
				in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
				matrix_pose, layer_type, input_img_size, output_img_size, hidden_dim=None, kernel_size=None, parameter_sharing='headwise',
				dropout = 0.):
		super().__init__()
  
		self.in_d_capsules = in_d_capsules
		self.out_d_capsules = out_d_capsules
		self.in_n_capsules = in_n_capsules
		self.out_n_capsules = out_n_capsules
		
		self.pose_dim = in_d_capsules
		self.layer_type = layer_type
		self.kernel_size = kernel_size
		self.matrix_pose = matrix_pose
		self.parameter_sharing = parameter_sharing

		if self.layer_type == 'FC':
			self.kernel_size=1

		if matrix_pose:
			# Random Initialisation of Two matrices
			self.matrix_pose_dim = int(np.sqrt(self.in_d_capsules))
			
			# w_current =(3,3,32,4,4)
			self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
													 in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
			self.w_next = nn.Parameter(0.02*torch.randn(
													 out_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
		else:
			self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
													 in_n_capsules, self.pose_dim, self.pose_dim))
			self.w_next = nn.Parameter(0.02*torch.randn(
													 out_n_capsules, self.pose_dim, self.pose_dim))

		
		max_seq_len = self.kernel_size*self.kernel_size*self.in_n_capsules
		heads = 1
		

		# Positional embeddings: 2 embeddings (1,7,1,8) and (1,1,7,8)
		self.rel_embedd_h = nn.Parameter(torch.randn(1, output_img_size,1, self.out_d_capsules //2), requires_grad=True)
		self.rel_embedd_w = nn.Parameter(torch.randn(1, 1, output_img_size, self.out_d_capsules //2), requires_grad=True)

		# self.rel_embedd = None
		self.dropout = nn.Dropout(dropout)
		print("You are using Bilinear routing with Linformer")


	def forward(self, current_pose, h_out=1, w_out=1, next_pose=None):
		
		# current pose: (b,32,3,3,7,7,16)
		# if FC current pose is (b, numcaps*h_in*w_in, caps_dim)
		if next_pose is None:
			# ist iteration
			batch_size = current_pose.shape[0]
			if self.layer_type=='conv':
				# (b, h_out, w_out, num_capsules, kernel_size, kernel_size, capsule_dim)
				# (b,7,7,32,3,3,16)
				current_pose = current_pose.permute([0,4,5,1,2,3,6])
				h_out = h_out
				w_out = w_out
			
			elif self.layer_type=='FC':
				h_out = 1
				w_out = 1
			pose_dim = self.pose_dim
			w_current = self.w_current
			w_next = self.w_next
			if self.matrix_pose:
				#w_current =(3,3,32,4,4) --> (3*3*32, 4, 4)
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim)
			
			#
			# W_current is C_{L} and w_next is N_{L}
			w_current = w_current.unsqueeze(0)  
			w_next = w_next.unsqueeze(0)

			current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)#view error
			
			if self.matrix_pose:
				# (b*7*7, 3*3*32, 4, 4) = (49b, 288, 4, 4)
				# print(current_pose.shape)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
			else:
				current_pose = current_pose.unsqueeze(2)
			
			# Multiplying p{L} by C_{L} to change to c_{L}
			# Current pose: (49b, 288, 4, 4), w_current = (1, 288, 4, 4)
			# Same matrix for the entire batch, output  = (49b, 288, 4, 4)
			current_pose = torch.matmul(current_pose, w_current) 
			

			
			if self.matrix_pose:
				# Current_pose = (49b, 288, 16)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
			else:
				current_pose = current_pose.squeeze(2)
			
			# R_{i,j} = (49b, m, 288)
			dots=(torch.ones(batch_size*h_out*w_out, self.out_n_capsules, self.kernel_size*self.kernel_size*self.in_n_capsules)* (pose_dim ** -0.5)).type_as(current_pose).to(current_pose)
			dots = dots.softmax(dim=-2)
			
 
			next_pose_candidates = current_pose  
			# Multiplies r_{i,j} with c_{L} ( no sorting in the 1st iteration) to give X. Still have to
			# multiply with N_{L} 
			# next pose: (49b, m, 16) 
			next_pose_candidates = torch.einsum('bij,bje->bie', dots, next_pose_candidates)
			
			###################### Positional Embeddings
			# (49b,m,16) --> (b,m,7,7,16) + rel_embedding (7,7,16) and then reshaped to (49b,m,16)
			# next_pose_candidates = next_pose_candidates.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
			# # next_pose_candidates = next_pose_candidates + self.rel_embedd
			# next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4)
			# next_pose_candidates = next_pose_candidates.reshape(-1,next_pose_candidates.shape[3], next_pose_candidates.shape[4])
			
			
			###################### Positional Embeddings in the middle
			# (49b,m,16) --> (b,m,7,7,16)
			next_pose_candidates = next_pose_candidates.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
			next_pose_candidates_h, next_pose_candidates_w = next_pose_candidates.split(self.pose_dim // 2, dim=4) # (b,m,7,7,8) and (b,m,7,7,8)
			# adding and concatenating (1,7,1,8) and (1,1,7,8) to (b,m,7,7,16)
			next_pose_candidates = torch.cat((next_pose_candidates_h + self.rel_embedd_h, next_pose_candidates_w + self.rel_embedd_w), dim=4)
			# next_pose_candidates = next_pose_candidates+self.rel_embedd
			next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4) # (b,7,7,m,16)
			next_pose_candidates = next_pose_candidates.reshape(-1,next_pose_candidates.shape[3], next_pose_candidates.shape[4])


			
			if self.matrix_pose:
				# Correct shapes: (49b, m, 4, 4)
				next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1], self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose_candidates = next_pose_candidates.unsqueeze(2)
			
			# Found final pose of next layer by multiplying X with N_{L}
			# Multiply (49b, m, 4, 4) with (1, m, 4, 4) == (49b, m , 4, 4)
			next_pose_candidates = torch.matmul(next_pose_candidates, w_next)

			# Reshape: (b, 7, 7, m, 16)
			next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
			
			
			###################### Positional Embeddings in the end
			# next_pose_candidates = next_pose_candidates.permute(0,3,1,2,4) #(b,m,7,7,16) 
			# next_pose_candidates_h, next_pose_candidates_w = next_pose_candidates.split(self.pose_dim // 2, dim=4) # (b,m,7,7,8) and (b,m,7,7,8)
			# # adding and concatenating (1,7,1,8) and (1,1,7,8) to (b,m,7,7,8)
			# next_pose_candidates = torch.cat((next_pose_candidates_h + self.rel_embedd_h, next_pose_candidates_w + self.rel_embedd_w), dim=4)
			# # next_pose_candidates = next_pose_candidates+self.rel_embedd
			# next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4)

			

			if self.layer_type == 'conv':
				# Reshape: (b,m,7,7,16) (just like original input, without expansion)
				next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
			
			elif self.layer_type == 'FC':
				# Reshape: (b, 1, 1, m, 16) --> (b, 1, m, 16) (h_out, w_out ==1)
				next_pose_candidates = next_pose_candidates.squeeze(1)
			return next_pose_candidates
		

		else:
			# 2nd to T iterations
			batch_size = next_pose.shape[0]
			if self.layer_type=='conv':
				# Current_pose = (b,7,7,32,3,3,16)
				current_pose = current_pose.permute([0,4,5,1,2,3,6])
				
				# next_pose = (b,m,7,7,16) --> (b,7,7,m,16)
				next_pose = next_pose.permute([0,2,3,1,4])
				h_out = next_pose.shape[1]
				w_out = next_pose.shape[2]
		   
			elif self.layer_type=='FC':
				h_out = 1
				w_out = 1
			
			pose_dim = self.pose_dim
			w_current = self.w_current
			w_next = self.w_next
			if self.matrix_pose:
				# w_current = (288,4,4)
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim) 
			
			# w_current = (1,288,4,4)
			w_current = w_current.unsqueeze(0)  
			w_next = w_next.unsqueeze(0)
			
			
			current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)            
			if self.matrix_pose:
				# Current_pose = (49b, 288, 4, 4)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
			else:
				current_pose = current_pose.unsqueeze(2)
			
			# Tranformed currentlayer capsules to c_{L}
			# Multiply (49b, 288, 4, 4) with (1,288,4,4) --> (49b, 288, 4, 4)
			current_pose = torch.matmul(current_pose, w_current)
			
			if self.matrix_pose:
				# Current_pose = (49b, 288, 16)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
			else:
				current_pose = current_pose.squeeze(2)

			
			###################### Positonal Embeddings
			# Adding positional embeddings to next pose: (b,7,7,m,16) -->(b,m,7,7,16)+(7,7,16)
			# print("original ", next_pose.shape)
			# next_pose = next_pose.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
			# print(next_pose.shape, self.rel_embedd.shape)
			# next_pose = next_pose + self.rel_embedd
				

			
			###################### Positional Embeddings in the Middle
			# (b,m,7,7,16)
			next_pose = next_pose.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
			next_pose_h, next_pose_w = next_pose.split(self.pose_dim // 2, dim=4) # (b,m,7,7,8) and (b,m,7,7,8)
			# adding and concatenating (1,7,1,8) and (1,1,7,8) to (b,m,7,7,16)
			next_pose = torch.cat((next_pose_h + self.rel_embedd_h, next_pose_w + self.rel_embedd_w), dim=4)
			# next_pose = next_pose+self.rel_embedd
			next_pose = next_pose.permute(0,2,3,1,4) # (b,7,7,m,16)
			next_pose = next_pose.reshape(-1,next_pose.shape[3], next_pose.shape[4])



			# next_pose = (b,m,7,7,16) --> (49b,m,16)   
			next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
			
			if self.matrix_pose:
				# next_pose = (49b,m,16)  -->  (49b,m,4,4) 
				next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose = next_pose.unsqueeze(3)
			
			# Tranform next pose using N_{L}: w_next = (49b,m,4,4) * (1,m,4,4)
			next_pose = torch.matmul(w_next, next_pose)
			

			if self.matrix_pose:
				# next_pose = (49b,m,16)
				next_pose = next_pose.view(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
			else:
				next_pose = next_pose.squeeze(3)
	
			# Finding scaled alignment scores between updated buckets
			# dots = (49b, m ,288)
			dots = torch.einsum('bje,bie->bji', next_pose, current_pose) * (pose_dim ** -0.5) 
			

			# attention routing along dim=-2 (next layer buckets)
			# Dim=-1 if you wanna invert the inverted attention
			dots = dots.softmax(dim=-2) 
			next_pose_candidates = current_pose

			# Yet to multiply with N_{L} (next_w)
			next_pose_candidates = torch.einsum('bji,bie->bje', dots, next_pose_candidates)
			
			if self.matrix_pose:
				# next pose: 49b,m,16 --> 49b,m,4,4
				next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1],self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose_candidates = next_pose_candidates.unsqueeze(3)
			
			# Multiplied with N_{j} to get final pose
			# w_next: (49b,m,4,4); b_next_pose_candidates: (49b,m , 4, 4)
			next_pose_candidates = torch.matmul(next_pose_candidates, w_next)
			
			# next_pose_candidates = (b,7,7,m,16)
			next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
			

			###################### Positional Embeddings in the end
			# next_pose_candidates = next_pose_candidates.permute(0,3,1,2,4) #(b,m,7,7,16) 
			# next_pose_candidates_h, next_pose_candidates_w = next_pose_candidates.split(self.pose_dim // 2, dim=4) # (b,m,7,7,8) and (b,m,7,7,8)
			# # adding and concatenating (1,7,1,8) and (1,1,7,8) to (b,m,7,7,8)
			# next_pose_candidates = torch.cat((next_pose_candidates_h + self.rel_embedd_h, next_pose_candidates_w + self.rel_embedd_w), dim=4)
			# # next_pose_candidates = next_pose_candidates+self.rel_embedd
			# next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4)

			


			if self.layer_type == 'conv':
				# next_pose_candidates = (b,m,7,7,16)
				next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
			elif self.layer_type == 'FC':
				# next_pose_candidates = (b,1,1,m,16) --> (b,1,m,16)
				next_pose_candidates = next_pose_candidates.squeeze(1)
			return next_pose_candidates


class PositionalEmbedding(nn.Module):
	"""
	Standard positional embedding.
	From the paper "Attention is all you need".
	Changed the constant from 10k to 100k, since this may be better for longer sequence lengths.
	"""
	def __init__(self, channels):
		super(PositionalEmbedding, self).__init__()
		inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
		self.register_buffer('inv_freq', inv_freq)

	def forward(self, tensor):
		pos = torch.arange(tensor.shape[1], device=tensor.device).type(self.inv_freq.type())
		sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
		emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
		return emb[None,:,:]


class LinformerProjectionKernelSinosoidEmbedding(nn.Module):
	def __init__(self, 
				in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
				matrix_pose, layer_type, input_img_size, output_img_size, hidden_dim=None, kernel_size=None, parameter_sharing='headwise',
				dropout = 0.):
		super().__init__()
  
		self.in_d_capsules = in_d_capsules
		self.out_d_capsules = out_d_capsules
		self.in_n_capsules = in_n_capsules
		self.out_n_capsules = out_n_capsules
		self.input_img_size=input_img_size
		self.output_img_size=output_img_size
		self.hidden_dim=hidden_dim
		self.pose_dim = in_d_capsules
		self.layer_type = layer_type
		self.kernel_size = kernel_size
		self.matrix_pose = matrix_pose
		self.parameter_sharing = parameter_sharing

		if self.layer_type == 'FC':
			self.kernel_size=1

		if matrix_pose:
			# Random Initialisation of Two matrices
			self.matrix_pose_dim = int(np.sqrt(self.in_d_capsules))
			
			# w_current =(3,3,32,4,4)
			self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
													 in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
			self.w_next = nn.Parameter(0.02*torch.randn(
													 out_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
		else:
			self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
													 in_n_capsules, self.pose_dim, self.pose_dim))
			self.w_next = nn.Parameter(0.02*torch.randn(
													 out_n_capsules, self.pose_dim, self.pose_dim))

		
		max_seq_len = self.kernel_size*self.kernel_size*self.in_n_capsules
		heads = 1
		
		if parameter_sharing == "headwise":
			# print("Hello")
			self.E_proj = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
													 in_n_capsules, hidden_dim))
	
		else:
			assert (False),"Yet to write the non-headwise method"
		

		self.pos_emb = PositionalEmbedding(self.pose_dim)

		self.dropout = nn.Dropout(dropout)
		print("You are using Bilinear routing with Linformer")


	def forward(self, current_pose, h_out=1, w_out=1, next_pose=None):
		
		# print('Using linformer kernels')
		# current pose: (b,32,3,3,7,7,16)
		# if FC current pose is (b, numcaps*h_in*w_in, caps_dim)
		if next_pose is None:
			# ist iteration
			batch_size = current_pose.shape[0]
			if self.layer_type=='conv':
				# (b, h_out, w_out, num_capsules, kernel_size, kernel_size, capsule_dim)
				# (b,7,7,32,3,3,16)
				current_pose = current_pose.permute([0,4,5,1,2,3,6])
				h_out = h_out
				w_out = w_out
			
			elif self.layer_type=='FC':
				h_out = 1
				w_out = 1

			pose_dim = self.pose_dim
			w_current = self.w_current
			w_next = self.w_next
			if self.matrix_pose:
				#w_current =(3,3,32,4,4) --> (3*3*32, 4, 4)
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim)
			
			#
			# W_current is C_{L} and w_next is N_{L}
			w_current = w_current.unsqueeze(0)  
			w_next = w_next.unsqueeze(0)

			current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)#view error
			
			if self.matrix_pose:
				# (b*7*7, 3*3*32, 4, 4) = (49b, 288, 4, 4)
				# print(current_pose.shape)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
			else:
				current_pose = current_pose.unsqueeze(2)
			
			# Multiplying p{L} by C_{L} to change to c_{L}
			# Current pose: (49b, 288, 4, 4), w_current = (1, 288, 4, 4)
			# Same matrix for the entire batch, output  = (49b, 288, 4, 4)
			current_pose = torch.matmul(current_pose, w_current) 


			if self.matrix_pose:
				# Current_pose = (49b, 288, 16)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
			else:
				current_pose = current_pose.squeeze(2)
			
			############## Linformer Projection
			current_pose = current_pose.permute(2,0,1) # (16,49b,288)
			E_proj = self.E_proj.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.hidden_dim) # (288, hidden_dim)            

			current_pose = torch.matmul(current_pose, E_proj) # (16,49b,hidden_dim)
			current_pose = current_pose.permute(1,2,0) # (49b, hidden_Dim, 16)

	
			# R_{i,j} = (49b, m, 288)
			dots=(torch.ones(batch_size*h_out*w_out, self.out_n_capsules, self.hidden_dim)* (pose_dim ** -0.5)).type_as(current_pose).to(current_pose)
			dots = dots.softmax(dim=-2)
			
 
			next_pose_candidates = current_pose  
			# Multiplies r_{i,j} with c_{L} ( no sorting in the 1st iteration) to give X. Still have to
			# multiply with N_{L} 
			# next pose: (49b, m, 16) 
			next_pose_candidates = torch.einsum('bij,bje->bie', dots, next_pose_candidates)
			
			###################### Positional Embeddings
			# (49b,m,16) --> (b,m,7,7,16) + rel_embedding (7,7,16) and then reshaped to (49b,m,16)
			next_pose_candidates = next_pose_candidates.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
			# next_pose_candidates = next_pose_candidates + self.rel_embedd
			next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4)
			next_pose_candidates = next_pose_candidates.reshape(-1,next_pose_candidates.shape[3], next_pose_candidates.shape[4])
			
			if self.matrix_pose:
				# Correct shapes: (49b, m, 4, 4)
				next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1], self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose_candidates = next_pose_candidates.unsqueeze(2)
			
			# Found final pose of next layer by multiplying X with N_{L}
			# Multiply (49b, m, 4, 4) with (1, m, 4, 4) == (49b, m , 4, 4)
			next_pose_candidates = torch.matmul(next_pose_candidates, w_next)

			# Reshape: (b, 7, 7, m, 16)
			next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
			
		

			if self.layer_type == 'conv':
				# Reshape: (b,m,7,7,16) (just like original input, without expansion)
				next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
			
			elif self.layer_type == 'FC':
				# Reshape: (b, 1, 1, m, 16) --> (b, 1, m, 16) (h_out, w_out ==1)
				next_pose_candidates = next_pose_candidates.squeeze(1)
			return next_pose_candidates
		

		else:
			# 2nd to T iterations
			batch_size = next_pose.shape[0]
			if self.layer_type=='conv':
				# Current_pose = (b,7,7,32,3,3,16)
				current_pose = current_pose.permute([0,4,5,1,2,3,6])
				
				# next_pose = (b,m,7,7,16) --> (b,7,7,m,16)
				next_pose = next_pose.permute([0,2,3,1,4])
				h_out = next_pose.shape[1]
				w_out = next_pose.shape[2]
		   
			elif self.layer_type=='FC':
				h_out = 1
				w_out = 1
			
			pose_dim = self.pose_dim
			w_current = self.w_current
			w_next = self.w_next
			if self.matrix_pose:
				# w_current = (288,4,4)
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim) 
			
			# w_current = (1,288,4,4)
			w_current = w_current.unsqueeze(0)  
			w_next = w_next.unsqueeze(0)
			
			
			current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)            
			if self.matrix_pose:
				# Current_pose = (49b, 288, 4, 4)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
			else:
				current_pose = current_pose.unsqueeze(2)
			
			# Tranformed currentlayer capsules to c_{L}
			# Multiply (49b, 288, 4, 4) with (1,288,4,4) --> (49b, 288, 4, 4)
			current_pose = torch.matmul(current_pose, w_current)
			
			if self.matrix_pose:
				# Current_pose = (49b, 288, 16)
				current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
			else:
				current_pose = current_pose.squeeze(2)

			
			############## Linformer Projection
			current_pose = current_pose.permute(2,0,1) # (16,49b,288)
			E_proj = self.E_proj.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.hidden_dim) # (288, hidden_dim)            

			current_pose = torch.matmul(current_pose, E_proj) # (16,49b,hidden_dim)
			current_pose = current_pose.permute(1,2,0) # (49b, hidden_Dim, 16)


			###################### Positonal Embeddings
			# Adding positional embeddings to next pose: (b,7,7,m,16) -->(b,m,7,7,16)+(7,7,16)
			# print("original ", next_pose.shape)
			next_pose = next_pose.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
			# print(next_pose.shape, self.rel_embedd.shape)
			# next_pose = next_pose + self.rel_embedd
				

			# next_pose = (b,m,7,7,16) --> (49b,m,16)   
			next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
			
			if self.matrix_pose:
				# next_pose = (49b,m,16)  -->  (49b,m,4,4) 
				next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose = next_pose.unsqueeze(3)
			
			# Tranform next pose using N_{L}: w_next = (49b,m,4,4) * (1,m,4,4)
			next_pose = torch.matmul(w_next, next_pose)
			

			if self.matrix_pose:
				# next_pose = (49b,m,16)
				next_pose = next_pose.view(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
			else:
				next_pose = next_pose.squeeze(3)
	
			# Finding scaled alignment scores between updated buckets
			# dots = (49b, m ,288)
			dots = torch.einsum('bje,bie->bji', next_pose, current_pose) * (pose_dim ** -0.5) 
			

			# attention routing along dim=-2 (next layer buckets)
			# Dim=-1 if you wanna invert the inverted attention
			dots = dots.softmax(dim=-2) 
			next_pose_candidates = current_pose

			# Yet to multiply with N_{L} (next_w)
			next_pose_candidates = torch.einsum('bji,bie->bje', dots, next_pose_candidates)
			
			if self.matrix_pose:
				# next pose: 49b,m,16 --> 49b,m,4,4
				next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1],self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose_candidates = next_pose_candidates.unsqueeze(3)
			
			# Multiplied with N_{j} to get final pose
			# w_next: (49b,m,4,4); b_next_pose_candidates: (49b,m , 4, 4)
			next_pose_candidates = torch.matmul(next_pose_candidates, w_next)
			
			# next_pose_candidates = (b,7,7,m,16)
			next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
			

			if self.layer_type == 'conv':
				# next_pose_candidates = (b,m,7,7,16)
				next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
			elif self.layer_type == 'FC':
				# next_pose_candidates = (b,1,1,m,16) --> (b,1,m,16)
				next_pose_candidates = next_pose_candidates.squeeze(1)
			return next_pose_candidates











# Global linformer Only FC type layers
#### Bilinear Linformer Capsule Layer ####
class BilinearGlobalLinformerRandomInitCapsuleFC(nn.Module):
	r"""Applies as a capsule fully-connected layer.
	TBD
	"""

	def __init__(self, hidden_dim, input_size, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, h_out, child_kernel_size, child_stride, child_padding, parent_kernel_size, parent_stride, parent_padding, parameter_sharing, matrix_pose, dp):

		super(BilinearGlobalLinformerCapsuleFC, self).__init__()
		self.in_n_capsules = in_n_capsules
		self.in_d_capsules = in_d_capsules
		self.out_n_capsules = out_n_capsules
		self.out_d_capsules = out_d_capsules
		self.h_out=h_out
		self.matrix_pose = matrix_pose
		self.hidden_dim =hidden_dim
		self.parameter_sharing=parameter_sharing
		self.pose_dim = in_d_capsules
		
		self.current_grouped_conv = nn.Conv2d(in_channels=self.in_n_capsules*in_d_capsules,
									 out_channels=self.in_n_capsules*in_d_capsules,
									 kernel_size=child_kernel_size,
									 stride=child_stride,
									 padding=child_padding,
									 groups= self.in_n_capsules,
									 bias=False)

		self.next_grouped_conv = nn.Conv2d(in_channels=out_n_capsules*out_d_capsules,
									 out_channels=out_n_capsules*out_d_capsules,
									 kernel_size= parent_kernel_size ,
									 stride=parent_stride,
									 padding=parent_padding,
									 groups= self.in_n_capsules,
									 bias=False)
		
		
		BilinearOutImg_size = int((input_size - child_kernel_size + 2* child_padding)/child_stride)+1
			
		if parameter_sharing == "headwise":
			self.E_proj = nn.Parameter(0.02*torch.randn(self.in_n_capsules, BilinearOutImg_size * BilinearOutImg_size, self.hidden_dim))

		else:
			# Correct this
			self.E_proj = nn.Parameter(0.02*torch.randn(self.in_n_capsules, self.in_d_capsules, 
									BilinearOutImg_size, BilinearOutImg_size))
  

		# Initiasation of gaussians for parent capsules
		self.mean = nn.Parameter(torch.randn(self.pose_dim))
		self.stdev = nn.Parameter(torch.randn(self.pose_dim))


		# Positional embedding
		# (256, 49)
		# self.rel_embedd = nn.Parameter(torch.randn(self.in_d_capsules, self.h_out , self.h_out), requires_grad=True)

		self.dropout_rate = dp
		self.nonlinear_act = nn.LayerNorm(out_d_capsules)
		self.drop = nn.Dropout(self.dropout_rate)
		self.scale = 1. / (out_d_capsules ** 0.5)

	

	def extra_repr(self):
		return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, matrix_pose={}, \
			weight_init_const={}, dropout_rate={}'.format(
			self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, self.matrix_pose,
			self.weight_init_const, self.dropout_rate
		)        
	def forward(self, current_pose, num_iter, next_pose = None):
		# current_pose shape: (b, num_Capsules, height, width, caps_dim), eg (b,32,16,16,256)
		import time
		time1=time.time()
		h_out=self.h_out
		w_out=h_out
		# rel_embedd = self.rel_embedd
		input_height, input_width = current_pose.shape[2], current_pose.shape[3]
		batch_size = current_pose.shape[0]
		pose_dim=self.in_d_capsules
		
		# Applying grouped convolution across capsule dimension
		if len(current_pose.shape) == 5:
			current_pose = current_pose.permute(0,2,3,1,4)
			current_pose = current_pose.contiguous().view(current_pose.shape[0], current_pose.shape[1], current_pose.shape[2],-1)
			current_pose = current_pose.permute(0,3,1,2)

		# current_pose : (b, 32*256, 16, 16) --> (b, 32*256, a, b) --> (b,32*a*b,256) == (b,n,d) in sequences
		time2 =time.time()
		current_pose = self.current_grouped_conv(current_pose)
		time3 =time.time()
		if self.parameter_sharing == 'headwise':
			E_proj = self.E_proj
			# Current pose becomes (b,256,32,1,a*b) 
			current_pose = current_pose.reshape(current_pose.shape[0], self.in_d_capsules,self.in_n_capsules,1, current_pose.shape[2] * current_pose.shape[3] )
			# current_pose - (b,256,32,1,a*b), E_proj  = (32,a*b,48) 
			# (b,256,32,1,48) --> (b,256,32,48)
			k_val = torch.matmul(current_pose, E_proj).squeeze()
			# reshaped to (b,256, hidden_dim*32)
			k_val = k_val.reshape(current_pose.shape[0], pose_dim, -1)
			k_val = k_val.permute(0,2,1)
			# print("Shape of k val is", k_val.shape)
			new_n_capsules = k_val.shape[1]
		
			time4 =time.time()
		
		else:
			assert (False), "write this method"
			# Correct this
			current_pose = current_pose.reshape(current_pose.shape[0], self.in_d_capsules, -1)
			current_pose = current_pose.permute(0,2,1)        
			transposed = torch.transpose(current_pose, 1, 2)
			k_val = self.E_proj(transposed)
			k_val = k_val.permute(0,2,1)
			# shape: (b, new_num_capsules, caps_dim)
			new_n_capsules = int(k_val.shape[1])
		
		# print(time2-time1, ' ', time3-time2,' ', time4-time3)
		
		if next_pose is None:
			batch_size = current_pose.shape[0]

			# Initialise next pose using gaussian distribution (b,m,7,7,16)
			mean=self.mean # 16
			stdev = self.stdev # 16
			mean = mean.view(1,1,1,1,-1) #(1,1,1,1,16)
			mean = mean.expand(batch_size,self.out_n_capsules,h_out, w_out, self.pose_dim) #(b,m,7,7,16)
			next_pose = torch.normal(mean, stdev)



			# dots = (torch.ones(batch_size, self.out_n_capsules*h_out*w_out, new_n_capsules)* (pose_dim ** -0.5)).type_as(k_val).to(k_val)
			# dots = F.softmax(dots, dim=-2)            
			# next_pose = torch.einsum('bji,bie->bje', dots, k_val)
			# # Adding relative positioning (caps_dim, h_out, w_out)
			# next_pose= next_pose.reshape(next_pose.shape[0],self.out_n_capsules, h_out,w_out, -1)
			# # rel_embedd  = rel_embedd.permute(1, 2, 0) # adding (b,m,h_out,w_out,caps_dim) and (h_out,w_out,caps_dim)
			# # next_pose = next_pose + rel_embedd

			# next_pose= next_pose.reshape(next_pose.shape[0],self.out_n_capsules * pose_dim,h_out,w_out)
			# next_pose = self.next_grouped_conv(next_pose)
		
			# next_pose =next_pose.reshape(batch_size, self.out_n_capsules, h_out, w_out, pose_dim)
			# return next_pose
		
		else:
			# next pose: (b,m,h_out,w_out,out_caps_dim) --> (b, m*out_cap_dim, h_out, w_out)
			h_out = next_pose.shape[2]
			w_out = next_pose.shape[3]
			
			# Adding relative positioning (caps_dim, h_out, w_out)
			# rel_embedd  = rel_embedd.permute(1, 2, 0)
			# next_pose = next_pose + rel_embedd # adding (b,m,h_out,w_out,caps_dim) and (h_out,w_out,caps_dim)
			# print("added rel embedding")
			next_pose = next_pose.permute(0,2,3,1,4)
			next_pose = next_pose.contiguous().view(next_pose.shape[0], next_pose.shape[1], next_pose.shape[2],-1)
			next_pose = next_pose.permute(0,3,1,2)

			next_pose = self.next_grouped_conv(next_pose)

			# (b, m*out_cap_dim, a, b) --> (b, m*a*b, out_caps_dim)
			next_pose = next_pose.reshape(next_pose.shape[0], self.out_n_capsules * next_pose.shape[2] * next_pose.shape[3], self.out_d_capsules)

			dots = torch.einsum('bje,bie->bji', next_pose, k_val) * (pose_dim ** -0.5) 
			dots = dots.softmax(dim=-2) 
			next_pose = torch.einsum('bji,bie->bje', dots, k_val)
			next_pose= next_pose.reshape(next_pose.shape[0],self.out_n_capsules * pose_dim,h_out,w_out)
			next_pose = self.next_grouped_conv(next_pose)

			# (b, m , a, b, out_caps_dim)
			next_pose =next_pose.reshape(batch_size, self.out_n_capsules, h_out, w_out, pose_dim)


		# Apply dropout
		next_pose = self.drop(next_pose)
		if not next_pose.shape[-1] == 1:
			if self.matrix_pose:
				next_pose = next_pose.view(next_pose.shape[0], 
									   next_pose.shape[1], self.out_d_capsules)
				next_pose = self.nonlinear_act(next_pose)
			else:
				next_pose = self.nonlinear_act(next_pose)
		return next_pose






# Global linformer Only FC type layers
#### Bilinear Linformer Capsule Layer ####
class BilinearGlobalLinformerRelativeEmbeddingsCapsuleFC(nn.Module):
	r"""Applies as a capsule fully-connected layer.
	TBD
	"""

	def __init__(self, hidden_dim, input_size, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, h_out, child_kernel_size, child_stride, child_padding, parent_kernel_size, parent_stride, parent_padding, parameter_sharing, matrix_pose, dp):

		super(BilinearGlobalLinformerRelativeEmbeddingsCapsuleFC, self).__init__()
		self.in_n_capsules = in_n_capsules
		self.in_d_capsules = in_d_capsules
		self.out_n_capsules = out_n_capsules
		self.out_d_capsules = out_d_capsules
		self.h_out=h_out
		self.matrix_pose = matrix_pose
		self.hidden_dim =hidden_dim
		self.parameter_sharing=parameter_sharing
		self.pose_dim = in_d_capsules
		
		self.current_grouped_conv = nn.Conv2d(in_channels=self.in_n_capsules*in_d_capsules,
									 out_channels=self.in_n_capsules*in_d_capsules,
									 kernel_size=child_kernel_size,
									 stride=child_stride,
									 padding=child_padding,
									 groups= self.in_n_capsules,
									 bias=False)

		self.next_grouped_conv = nn.Conv2d(in_channels=out_n_capsules*out_d_capsules,
									 out_channels=out_n_capsules*out_d_capsules,
									 kernel_size= parent_kernel_size ,
									 stride=parent_stride,
									 padding=parent_padding,
									 groups= self.in_n_capsules,
									 bias=False)
		
		
		BilinearOutImg_size = int((input_size - child_kernel_size + 2* child_padding)/child_stride)+1
			
		if parameter_sharing == "headwise":
			self.E_proj = nn.Parameter(0.02*torch.randn(self.in_n_capsules, BilinearOutImg_size * BilinearOutImg_size, self.hidden_dim))

		else:
			# Correct this
			self.E_proj = nn.Parameter(0.02*torch.randn(self.in_n_capsules, self.in_d_capsules, 
									BilinearOutImg_size, BilinearOutImg_size))
  

		# Positional embeddings: 2 embeddings (1,7,1,8) and (1,1,7,8)
		output_img_size = 1 #Global Linformers
		self.rel_embedd_h = nn.Parameter(torch.randn(1, output_img_size,1, self.out_d_capsules //2), requires_grad=True)
		self.rel_embedd_w = nn.Parameter(torch.randn(1, 1, output_img_size, self.out_d_capsules //2), requires_grad=True)



		# Positional embedding
		# (256, 49)
		# self.rel_embedd = nn.Parameter(torch.randn(self.in_d_capsules, self.h_out , self.h_out), requires_grad=True)

		self.dropout_rate = dp
		self.nonlinear_act = nn.LayerNorm(out_d_capsules)
		self.drop = nn.Dropout(self.dropout_rate)
		self.scale = 1. / (out_d_capsules ** 0.5)

	

	def extra_repr(self):
		return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, matrix_pose={}, \
			weight_init_const={}, dropout_rate={}'.format(
			self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, self.matrix_pose,
			self.weight_init_const, self.dropout_rate
		)        
	def forward(self, current_pose, num_iter, next_pose = None):
		# current_pose shape: (b, num_Capsules, height, width, caps_dim), eg (b,32,16,16,256)
		import time
		time1=time.time()
		h_out=self.h_out
		w_out=h_out
		# rel_embedd = self.rel_embedd
		input_height, input_width = current_pose.shape[2], current_pose.shape[3]
		batch_size = current_pose.shape[0]
		pose_dim=self.in_d_capsules
		
		# Applying grouped convolution across capsule dimension
		if len(current_pose.shape) == 5:
			current_pose = current_pose.permute(0,2,3,1,4)
			current_pose = current_pose.contiguous().view(current_pose.shape[0], current_pose.shape[1], current_pose.shape[2],-1)
			current_pose = current_pose.permute(0,3,1,2)

		# current_pose : (b, 32*256, 16, 16) --> (b, 32*256, a, b) --> (b,32*a*b,256) == (b,n,d) in sequences
		current_pose = self.current_grouped_conv(current_pose)
		if self.parameter_sharing == 'headwise':
			E_proj = self.E_proj
			# Current pose becomes (b,256,32,1,a*b) 
			current_pose = current_pose.reshape(current_pose.shape[0], self.in_d_capsules,self.in_n_capsules,1, current_pose.shape[2] * current_pose.shape[3] )
			
			# current_pose - (b,256,32,1,a*b), E_proj  = (32,a*b,48) 
			# (b,256,32,1,48) --> (b,256,32,48)
			k_val = torch.matmul(current_pose, E_proj).squeeze()
			# reshaped to (b,256, hidden_dim*32)
			k_val = k_val.reshape(current_pose.shape[0], pose_dim, -1)
			k_val = k_val.permute(0,2,1)
			# print("Shape of k val is", k_val.shape)
			new_n_capsules = k_val.shape[1]
		
			time4 =time.time()
		
		else:
			assert (False), "write this method"
			# Correct this
			current_pose = current_pose.reshape(current_pose.shape[0], self.in_d_capsules, -1)
			current_pose = current_pose.permute(0,2,1)        
			transposed = torch.transpose(current_pose, 1, 2)
			k_val = self.E_proj(transposed)
			k_val = k_val.permute(0,2,1)
			# shape: (b, new_num_capsules, caps_dim)
			new_n_capsules = int(k_val.shape[1])
		
		# print(time2-time1, ' ', time3-time2,' ', time4-time3)
		
		if next_pose is None:

			dots = (torch.ones(batch_size*h_out*w_out, self.out_n_capsules, new_n_capsules)* (pose_dim ** -0.5)).type_as(k_val).to(k_val)
			dots = F.softmax(dots, dim=-2)  
			dots = dots.reshape(batch_size, self.out_n_capsules*h_out*w_out, new_n_capsules)          
			next_pose = torch.einsum('bji,bie->bje', dots, k_val) # 
			
			# Adding relative positioning to (batch_size, self.out_n_capsules*h_out*w_out, pose_dim)
			# (b,m,h_out,w_out,pose_dim)
			next_pose= next_pose.reshape(next_pose.shape[0],self.out_n_capsules, h_out,w_out, -1)
			next_pose_h, next_pose_w = next_pose.split(self.pose_dim // 2, dim=4)
			next_pose = torch.cat((next_pose_h + self.rel_embedd_h, next_pose_w + self.rel_embedd_w), dim=4)
			

			next_pose= next_pose.reshape(next_pose.shape[0],self.out_n_capsules * pose_dim,h_out,w_out)
			next_pose = self.next_grouped_conv(next_pose)
		
			next_pose =next_pose.reshape(batch_size, self.out_n_capsules, h_out, w_out, pose_dim)
			return next_pose
		
		else:
			# next pose: (b,m,h_out,w_out,out_caps_dim) --> (b, m*out_cap_dim, h_out, w_out)
			h_out = next_pose.shape[2]
			w_out = next_pose.shape[3]
			

			# Adding relative positioning to (batch_size, self.out_n_capsules*h_out*w_out, pose_dim)
			# (b,m,h_out,w_out,pose_dim)
			next_pose= next_pose.reshape(next_pose.shape[0],self.out_n_capsules, h_out,w_out, -1)
			next_pose_h, next_pose_w = next_pose.split(self.pose_dim // 2, dim=4)
			next_pose = torch.cat((next_pose_h + self.rel_embedd_h, next_pose_w + self.rel_embedd_w), dim=4)
			

			# Adding relative positioning (caps_dim, h_out, w_out)
			# rel_embedd  = rel_embedd.permute(1, 2, 0)
			# next_pose = next_pose + rel_embedd # adding (b,m,h_out,w_out,caps_dim) and (h_out,w_out,caps_dim)
			# print("added rel embedding")
			next_pose = next_pose.permute(0,2,3,1,4)
			next_pose = next_pose.contiguous().view(next_pose.shape[0], next_pose.shape[1], next_pose.shape[2],-1)
			next_pose = next_pose.permute(0,3,1,2)

			next_pose = self.next_grouped_conv(next_pose)

			# (b, m*out_cap_dim, a, b) --> (b, m*a*b, out_caps_dim)
			next_pose = next_pose.reshape(next_pose.shape[0], self.out_n_capsules * next_pose.shape[2] * next_pose.shape[3], self.out_d_capsules)

			dots = torch.einsum('bje,bie->bji', next_pose, k_val) * (pose_dim ** -0.5) 
			dots = dots.softmax(dim=-2) 
			next_pose = torch.einsum('bji,bie->bje', dots, k_val)
			next_pose= next_pose.reshape(next_pose.shape[0],self.out_n_capsules * pose_dim,h_out,w_out)
			next_pose = self.next_grouped_conv(next_pose)

			# (b, m , a, b, out_caps_dim)
			next_pose =next_pose.reshape(batch_size, self.out_n_capsules, h_out, w_out, pose_dim)


		# Apply dropout
		next_pose = self.drop(next_pose)
		if not next_pose.shape[-1] == 1:
			if self.matrix_pose:
				next_pose = next_pose.view(next_pose.shape[0], 
									   next_pose.shape[1], self.out_d_capsules)
				next_pose = self.nonlinear_act(next_pose)
			else:
				next_pose = self.nonlinear_act(next_pose)
		return next_pose









# DEBUGG VERSION WHERE CONV = MAT MULT for transformation
# Global linformer Only FC type layers
#### Bilinear Linformer Capsule Layer ####
class BilinearGlobalLinformerDebugCapsuleFC(nn.Module):
	r"""Applies as a capsule fully-connected layer.
	TBD
	"""

	def __init__(self, hidden_dim, input_size, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, h_out, child_kernel_size, child_stride, child_padding, parent_kernel_size, parent_stride, parent_padding, parameter_sharing, matrix_pose, dp):

		super(BilinearGlobalLinformerDebugCapsuleFC, self).__init__()
		self.in_n_capsules = in_n_capsules
		self.in_d_capsules = in_d_capsules
		self.out_n_capsules = out_n_capsules
		self.out_d_capsules = out_d_capsules
		self.h_out=h_out
		self.matrix_pose = matrix_pose
		self.hidden_dim =hidden_dim
		self.parameter_sharing=parameter_sharing
		self.pose_dim = in_d_capsules        

		self.child_kernel_size = child_kernel_size
		self.child_stride = child_stride
		self.child_padding = child_padding

		self.parent_kernel_size = parent_kernel_size
		self.parent_stride = parent_stride
		self.parent_padding = parent_padding


		if matrix_pose:
			# Random Initialisation of Two matrices
			self.matrix_pose_dim = int(np.sqrt(self.in_d_capsules))
			
			# w_current =(1,1,32,4,4)
			self.w_current = nn.Parameter(0.02*torch.randn(child_kernel_size, child_kernel_size,
													 in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
			self.w_next = nn.Parameter(0.02*torch.randn(
													 out_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
		else:
			self.w_current = nn.Parameter(0.02*torch.randn(child_kernel_size, child_kernel_size,
													 in_n_capsules, self.pose_dim, self.pose_dim))
			self.w_next = nn.Parameter(0.02*torch.randn(
													 out_n_capsules, self.pose_dim, self.pose_dim))

		

		BilinearOutImg_size = int((input_size - child_kernel_size + 2* child_padding)/child_stride)+1
			
		if parameter_sharing == "headwise":
			self.E_proj = nn.Parameter(0.02*torch.randn(self.in_n_capsules, BilinearOutImg_size * BilinearOutImg_size, self.hidden_dim))

		else:
			assert (False), "Write the codefor non-headwise linformer"

		self.rel_embedd_h = nn.Parameter(torch.randn(1, self.h_out,1, self.out_d_capsules //2), requires_grad=True)
		self.rel_embedd_w = nn.Parameter(torch.randn(1, 1, self.h_out, self.out_d_capsules //2), requires_grad=True)



		self.dropout_rate = dp
		self.nonlinear_act = nn.LayerNorm(out_d_capsules)
		self.drop = nn.Dropout(self.dropout_rate)
		self.scale = 1. / (out_d_capsules ** 0.5)

	  
	def forward(self, current_pose, num_iter, next_pose = None):
		# current_pose shape: (b, num_Capsules, height, width, caps_dim), eg (b,32,16,16,4,4)

		h_out=self.h_out
		w_out=h_out
		w_current = self.w_current #(Hkernel, Wkernel, 32, 4, 4) == (1,1,32,4,4)
		w_next = self.w_next

		batch_size = current_pose.shape[0]
		pose_dim=self.in_d_capsules
		
		if current_pose.ndim == 5: # Last layer was conv capsule
			current_size = current_pose.shape[2]
		else:
			current_size=1


		######### Transforming input using matrix mul and w_current's     
		if self.matrix_pose:
			#w_current =(1,1,32,4,4) --> (1*1*32, 4, 4)
			w_current = w_current.view(self.child_kernel_size*self.child_kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
		else:
			w_current = w_current.view(self.child_kernel_size*self.child_kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim)
		
		w_current = w_current.unsqueeze(0)  # (1,1*1*32,4,4)
		w_next = w_next.unsqueeze(0) # (1,m,4,4)

		current_pose = current_pose.reshape(batch_size*current_size*current_size, self.child_kernel_size*self.child_kernel_size*self.in_n_capsules, self.pose_dim) #(16*16*b,1*1*32,256)

		if self.matrix_pose:
			current_pose = current_pose.reshape(batch_size*current_size*current_size, self.child_kernel_size*self.child_kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
		else:
			current_pose = current_pose.unsqueeze(2) # (16*16*b,1*1*32,1,16)
	
		# Current pose: (16*16*b,1*1*32,4,4), w_current = (1,1*1*32,4,4) --> (256b,32,4,4)
		current_pose = torch.matmul(current_pose, w_current) 
		
		if not self.matrix_pose:
			current_pose = current_pose.squeeze(2) #(256b,32,1,16)-->(256b,32,16)
		

		######### Linformer Projection
		current_pose = current_pose.reshape(batch_size, self.pose_dim, self.in_n_capsules, 1, -1) # (b,256,32,1,a*b) 
		if self.parameter_sharing == 'headwise':
			E_proj = self.E_proj        			
			# current_pose - (b,16,32,1,a*b), E_proj  = (32,a*b,48) 
			# print('final: ', current_pose.shape, ' E_proj ', E_proj.shape)
			k_val = torch.matmul(current_pose, E_proj).squeeze() # (b,16,32,1,48) --> (b,16,32,48)
			k_val = k_val.reshape(current_pose.shape[0], pose_dim, -1) # Reshaped to (b,16, hidden_dim*32) ; last value indicates new seq length
			k_val = k_val.permute(0,2,1) #(b,hidden_dim * num_Caps, pose_dim)
			new_n_capsules = k_val.shape[1]
				
		else:
			assert (False), "Write non-headwise linformer code"
		
		
		if next_pose is None:
			dots = (torch.ones(batch_size*h_out*w_out, self.out_n_capsules, new_n_capsules)* (pose_dim ** -0.5)).type_as(k_val).to(k_val)
			dots = F.softmax(dots, dim=-2)
			value = np.unique(dots.cpu().numpy())[0]
			next_pose_candidates = k_val   
			# print('dots: ', dots.shape, ' k_val ', k_val.shape)
			next_pose_candidates = torch.sum(next_pose_candidates, dim=1) * value
			next_pose_candidates = next_pose_candidates.unsqueeze(1)
			next_pose_candidates = next_pose_candidates.expand(next_pose_candidates.shape[0], self.out_n_capsules, next_pose_candidates.shape[2])

			# next_pose_candidates = torch.einsum('bji,bie->bje', dots, k_val) # Feasible iff h_out = 1 = w_out
			

			if self.matrix_pose:
				# Correct shapes: (1*1*b, m, 4, 4)
				next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1], self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose_candidates = next_pose_candidates.unsqueeze(2) #(b,m,1,16)
			
			next_pose_candidates = torch.matmul(next_pose_candidates, w_next)

			# Reshape: (b, 7, 7, m, 16)
			next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
			
			# Assuming the layer was definitely FC
			# (b,1,m,16)
			next_pose_candidates = next_pose_candidates.squeeze(1)

		
		else:
			###### Tranforming next layer capsule
			next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim) # (1*1*b,m,16)
			
			if self.matrix_pose:
				next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose = next_pose.unsqueeze(3) # (b, m, 16, 1)
			
			next_pose = torch.matmul(w_next, next_pose)

			if self.matrix_pose:
				next_pose = next_pose.view(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
			else:
				next_pose = next_pose.squeeze(3)

			dots = torch.einsum('bje,bie->bji', next_pose, k_val) * (pose_dim ** -0.5)  # Holds iff h_out=1
			dots = dots.softmax(dim=-2) 
			next_pose_candidates = torch.einsum('bji,bie->bje', dots, k_val)
			
			###### Transforming next pose final time for pose updation
			if self.matrix_pose:
				next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1],self.matrix_pose_dim, self.matrix_pose_dim)
			else:
				next_pose_candidates = next_pose_candidates.unsqueeze(3)
			
			next_pose_candidates = torch.matmul(w_next, next_pose_candidates)
			
			next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
			
			# # Assuming the layer was definitely FC
			# next_pose_candidates = (b,1,1,m,16) --> (b,1,m,16)
			next_pose_candidates = next_pose_candidates.squeeze(1)


		# Apply dropout
		next_pose_candidates = self.drop(next_pose_candidates)
		if not next_pose_candidates.shape[-1] == 1:
			next_pose_candidates = self.nonlinear_act(next_pose_candidates)

		return next_pose_candidates
























#Capsules Linformer projections but with convolution capsules too
# different tranformation for each patch 
# class LinformerProjectionEntireOutImg(nn.Module):
#     def __init__(self, 
#                 in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
#                 matrix_pose, layer_type, input_img_size, output_img_size, hidden_dim=None, kernel_size=None, parameter_sharing='headwise',
#                 dropout = 0.):
#         super().__init__()
  
#         self.in_d_capsules = in_d_capsules
#         self.out_d_capsules = out_d_capsules
#         self.in_n_capsules = in_n_capsules
#         self.out_n_capsules = out_n_capsules
#         self.input_img_size=input_img_size
#         self.output_img_size=output_img_size
#         self.hidden_dim=hidden_dim

#         self.pose_dim = in_d_capsules
#         self.layer_type = layer_type
#         self.kernel_size = kernel_size
#         self.matrix_pose = matrix_pose
#         self.parameter_sharing = parameter_sharing

#         if self.layer_type == 'FC':
#             self.kernel_size=1

#         if matrix_pose:
#             # Random Initialisation of Two matrices
#             self.matrix_pose_dim = int(np.sqrt(self.in_d_capsules))
			
#             # w_current =(3,3,32,4,4)
#             self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
#                                                      in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
#             self.w_next = nn.Parameter(0.02*torch.randn(
#                                                      out_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
#         else:
#             self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
#                                                      in_n_capsules, self.pose_dim, self.pose_dim))
#             self.w_next = nn.Parameter(0.02*torch.randn(
#                                                      out_n_capsules, self.pose_dim, self.pose_dim))

		
#         max_seq_len = self.kernel_size*self.kernel_size*self.in_n_capsules
#         heads = 1
		
#         if parameter_sharing == "headwise":
#             # print("Hello")
#             if self.layer_type =='conv':
#                 self.E_proj = nn.Parameter(0.02*torch.randn(self.in_n_capsules, output_img_size * output_img_size, hidden_dim))
#             else:
#                 self.E_proj = nn.Parameter(0.02*torch.randn(int(self.in_n_capsules/(self.input_img_size * self.input_img_size)), input_img_size * input_img_size, hidden_dim))
	
#         else:
#             assert (False),"Yet to write the non-headwise method"

#         # Positional embeddings: (7,7,16)
#         self.rel_embedd = nn.Parameter(torch.randn(output_img_size, output_img_size, self.out_d_capsules), requires_grad=True)
#         # self.rel_embedd = None
#         self.dropout = nn.Dropout(dropout)
#         print("You are using Bilinear routing with Linformer")


#     def forward(self, current_pose, h_out=1, w_out=1, next_pose=None):
		
#         # current pose: (b,32,3,3,7,7,16)
#         # if FC current pose is (b, numcaps*h_in*w_in, caps_dim)
#         if next_pose is None:
#             # ist iteration
#             batch_size = current_pose.shape[0]
#             if self.layer_type=='conv':
#                 # (b, h_out, w_out, num_capsules, kernel_size, kernel_size, capsule_dim)
#                 # (b,7,7,32,3,3,16)
#                 current_pose = current_pose.permute([0,4,5,1,2,3,6])
#                 h_out = h_out
#                 w_out = w_out
			
#             elif self.layer_type=='FC':
#                 h_out = 1
#                 w_out = 1
#             pose_dim = self.pose_dim
#             w_current = self.w_current
#             w_next = self.w_next
#             if self.matrix_pose:
#                 #w_current =(3,3,32,4,4) --> (3*3*32, 4, 4)
#                 w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
#             else:
#                 w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim)
			
#             #
#             # W_current is C_{L} and w_next is N_{L}
#             w_current = w_current.unsqueeze(0)  
#             w_next = w_next.unsqueeze(0)

#             current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)#view error
			
#             if self.matrix_pose:
#                 # (b*7*7, 3*3*32, 4, 4) = (49b, 288, 4, 4)
#                 # print(current_pose.shape)
#                 current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
#             else:
#                 current_pose = current_pose.unsqueeze(2)
			
#             # Multiplying p{L} by C_{L} to change to c_{L}
#             # Current pose: (49b, 288, 4, 4), w_current = (1, 288, 4, 4)
#             # Same matrix for the entire batch, output  = (49b, 288, 4, 4)
#             current_pose = torch.matmul(current_pose, w_current) 
			
			
#             if self.matrix_pose:
#                 # Current_pose = (49b, 288, 16)
#                 current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
#             else:
#                 current_pose = current_pose.squeeze(2)
			
			

#             # Linformer projection
#             # (b,3,3,16,32,1,7*7) X (32,49,hidden_dim) --> (b,3,3,16,32,1,hidden_dim)
#             if self.layer_type=='conv':
#                 current_pose = current_pose.reshape(batch_size, self.kernel_size, self.kernel_size , self.pose_dim, self.in_n_capsules, 1, h_out*w_out)
#                 # print("Input shape: ", current_pose.shape, self.E_proj.shape)
#                 current_pose = torch.matmul(current_pose, self.E_proj).squeeze(5)
#                 current_pose = current_pose.reshape(current_pose.shape[0], current_pose.shape[1]*current_pose.shape[2]*current_pose.shape[4]*current_pose.shape[5], current_pose.shape[3])
#                 dots=(torch.ones(batch_size, self.out_n_capsules*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules * self.hidden_dim)* (pose_dim ** -0.5)).type_as(current_pose).to(current_pose)

#             else:
#                 # Input is (b, num_Caps*input_size*input_size,16) ==(b,5*5*32,16) -> (b,16,32,1,5*5) X (32,25,hidden_dim) --> (b,16,32,1,hidden_dim) -> (b,32*hidden_dim, 16)
#                 current_pose = current_pose.reshape(batch_size, self.pose_dim, int(self.in_n_capsules/(self.input_img_size * self.input_img_size)), 1, self.input_img_size * self.input_img_size)  
#                 # print("Input shape: ", current_pose.shape, self.E_proj.shape)
#                 current_pose = torch.matmul(current_pose, self.E_proj).squeeze(3)
#                 current_pose = current_pose.reshape(current_pose.shape[0], current_pose.shape[2]*current_pose.shape[3], current_pose.shape[1])
#                 dots=(torch.ones(batch_size*h_out*w_out, self.out_n_capsules, self.kernel_size*self.kernel_size* int(self.in_n_capsules/(self.input_img_size * self.input_img_size)) * self.hidden_dim)* (pose_dim ** -0.5)).type_as(current_pose).to(current_pose)
#                 # print("Input shape: ", current_pose.shape, dots.shape)


#             # R_{i,j} = (b, m*7*7, 3*3*32*hidden_dim)
#             dots = dots.softmax(dim=-2)
			
 
#             next_pose_candidates = current_pose  
#             # Multiplies r_{i,j} with c_{L} ( no sorting in the 1st iteration) to give X. Still have to
#             # multiply with N_{L} 
#             # next pose: (49b, m, 16) 
#             next_pose_candidates = torch.einsum('bij,bje->bie', dots, next_pose_candidates)
#             # (49b,m,16) --> (b,m,7,7,16) + rel_embedding (7,7,16) and then reshaped to (49b,m,16)
#             next_pose_candidates = next_pose_candidates.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
			
#             next_pose_candidates = next_pose_candidates + self.rel_embedd
#             next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4)
#             next_pose_candidates = next_pose_candidates.reshape(-1,next_pose_candidates.shape[3], next_pose_candidates.shape[4])
#             if self.matrix_pose:
#                 # Correct shapes: (49b, m, 4, 4)
#                 next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1], self.matrix_pose_dim, self.matrix_pose_dim)
#             else:
#                 next_pose_candidates = next_pose_candidates.unsqueeze(2)
			
#             # Found final pose of next layer by multiplying X with N_{L}
#             # Multiply (49b, m, 4, 4) with (1, m, 4, 4) == (49b, m , 4, 4)
#             next_pose_candidates = torch.matmul(next_pose_candidates, w_next)

#             # Reshape: (b, 7, 7, m, 16)
#             next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
			
#             if self.layer_type == 'conv':
#                 # Reshape: (b,m,7,7,16) (just like original input, without expansion)
#                 next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
			
#             elif self.layer_type == 'FC':
#                 # Reshape: (b, 1, 1, m, 16) --> (b, 1, m, 16) (h_out, w_out ==1)
#                 next_pose_candidates = next_pose_candidates.squeeze(1)
#             return next_pose_candidates
		

#         else:
#             # 2nd to T iterations
#             batch_size = next_pose.shape[0]
#             if self.layer_type=='conv':
#                 # Current_pose = (b,7,7,32,3,3,16)
#                 current_pose = current_pose.permute([0,4,5,1,2,3,6])
				
#                 # next_pose = (b,m,7,7,16) --> (b,7,7,m,16)
#                 next_pose = next_pose.permute([0,2,3,1,4])
#                 h_out = next_pose.shape[1]
#                 w_out = next_pose.shape[2]
		   
#             elif self.layer_type=='FC':
#                 h_out = 1
#                 w_out = 1
			
#             pose_dim = self.pose_dim
#             w_current = self.w_current
#             w_next = self.w_next
#             if self.matrix_pose:
#                 # w_current = (288,4,4)
#                 w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
#             else:
#                 w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim) 
			
#             # w_current = (1,288,4,4)
#             w_current = w_current.unsqueeze(0)  
#             w_next = w_next.unsqueeze(0)
			
			
#             current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)            
#             if self.matrix_pose:
#                 # Current_pose = (49b, 288, 4, 4)
#                 current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
#             else:
#                 current_pose = current_pose.unsqueeze(2)
			
#             # Tranformed currentlayer capsules to c_{L}
#             # Multiply (49b, 288, 4, 4) with (1,288,4,4) --> (49b, 288, 4, 4)
#             current_pose = torch.matmul(current_pose, w_current)
			
#             if self.matrix_pose:
#                 # Current_pose = (49b, 288, 16)
#                 current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
#             else:
#                 current_pose = current_pose.squeeze(2)



#             # Linformer projection
#             # (b,3,3,16,32,1,7*7) X (32,49,hidden_dim) --> (b,3,3,16,32,1,hidden_dim)
#             if self.layer_type=='conv':
#                 # print(current_pose.shape)
#                 current_pose = current_pose.reshape(batch_size, self.kernel_size, self.kernel_size , self.pose_dim, self.in_n_capsules, 1, h_out*w_out)
#                 # print("Input shape: ", current_pose.shape, self.E_proj.shape)
#                 current_pose = torch.matmul(current_pose, self.E_proj).squeeze(5)
#                 current_pose = current_pose.reshape(current_pose.shape[0], current_pose.shape[1]*current_pose.shape[2]*current_pose.shape[4]*current_pose.shape[5], current_pose.shape[3])

#             else:
#                 # Input is (b, num_Caps*input_size*input_size,16) ==(b,5*5*32,16) -> (b,16,32,1,5*5) X (32,25,hidden_dim) --> (b,16,32,1,hidden_dim) -> (b,32*hidden_dim, 16)
#                 current_pose = current_pose.reshape(batch_size, self.pose_dim, int(self.in_n_capsules/(self.input_img_size * self.input_img_size)), 1, self.input_img_size * self.input_img_size)  
#                 # print("Input shape: ", current_pose.shape, self.E_proj.shape)
#                 current_pose = torch.matmul(current_pose, self.E_proj).squeeze(3)
#                 current_pose = current_pose.reshape(current_pose.shape[0], current_pose.shape[2]*current_pose.shape[3], current_pose.shape[1])



			
#             # Adding positional embeddings to next pose: (b,7,7,m,16) -->(b,m,7,7,16)+(7,7,16)
#             # print("original ", next_pose.shape)
#             next_pose = next_pose.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
#             # print(next_pose.shape, self.rel_embedd.shape)
#             next_pose = next_pose + self.rel_embedd
				


#             # next_pose = (b,m,7,7,16) --> (49b,m,16)   
#             next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
			
#             if self.matrix_pose:
#                 # next_pose = (49b,m,16)  -->  (49b,m,4,4) 
#                 next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.matrix_pose_dim, self.matrix_pose_dim)
#             else:
#                 next_pose = next_pose.unsqueeze(3)
			
#             # Tranform next pose using N_{L}: w_next = (49b,m,4,4) * (1,m,4,4)

#             next_pose = torch.matmul(w_next, next_pose)
			

#             if self.matrix_pose:
#                 # next_pose = (b,49m,16)
#                 if self.layer_type=='conv':
#                     next_pose = next_pose.view(batch_size, self.out_n_capsules*h_out*w_out,  self.pose_dim)
#                 else:
#                     next_pose = next_pose.view(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
				   
#             else:
#                 next_pose = next_pose.squeeze(3)
	
			
#             # Finding scaled alignment scores between updated buckets
#             # dots = (49b, m ,288*hidden_dim)
			
#             dots = torch.einsum('bje,bie->bji', next_pose, current_pose) * (pose_dim ** -0.5) 
#             # print("dots time shape: ", current_pose.shape, next_pose.shape, dots.shape)
			

#             # attention routing along dim=-2 (next layer buckets)
#             # Dim=-1 if you wanna invert the inverted attention
#             dots = dots.softmax(dim=-2) 
#             next_pose_candidates = current_pose

#             # Yet to multiply with N_{L} (next_w)

#             next_pose_candidates = torch.einsum('bji,bie->bje', dots, next_pose_candidates)
#             # print("Netx canditate: ", next_pose_candidates.shape)

#             if self.matrix_pose:
#                 # next pose: 49b,m,16 --> 49b,m,4,4
#                 next_pose_candidates=next_pose_candidates.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
#                 next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1],self.matrix_pose_dim, self.matrix_pose_dim)
#             else:
#                 next_pose_candidates = next_pose_candidates.unsqueeze(3)
			
#             # Multiplied with N_{j} to get final pose
#             # w_next: (49b,m,4,4); b_next_pose_candidates: (49b,m , 4, 4)
#             next_pose_candidates = torch.matmul(next_pose_candidates, w_next)
			
#             # next_pose_candidates = (b,7,7,m,16)
#             next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
			
#             if self.layer_type == 'conv':
#                 # next_pose_candidates = (b,m,7,7,16)
#                 next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
#             elif self.layer_type == 'FC':
#                 # next_pose_candidates = (b,1,1,m,16) --> (b,1,m,16)
#                 next_pose_candidates = next_pose_candidates.squeeze(1)
#             return next_pose_candidates










