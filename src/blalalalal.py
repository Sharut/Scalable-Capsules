# DYANMIC BILINEAR ROUTING
class SaraSabourAdaptedDynamicRouting(nn.Module):
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
                matrix_pose, layer_type, kernel_size=None, dropout = 0.):
        super().__init__()
        
        self.in_d_capsules = in_d_capsules
        self.out_d_capsules = out_d_capsules
        self.in_n_capsules = in_n_capsules
        self.out_n_capsules = out_n_capsules
        
        self.pose_dim = in_d_capsules
        self.layer_type = layer_type
        self.kernel_size = kernel_size
        self.matrix_pose = matrix_pose

        if self.layer_type == 'FC':
            self.kernel_size=1

        
        if matrix_pose:
            self.sqrt_d = int(np.sqrt(self.in_d_capsules))
            self.weight_init_const = np.sqrt(out_n_capsules/(self.sqrt_d*in_n_capsules)) 
            
            # (n,4,4,m)
            self.w = nn.Parameter(self.weight_init_const* \
                                          torch.randn(in_n_capsules, self.sqrt_d, self.sqrt_d, out_n_capsules))
        
        # Vector form of Hilton  
        else:
            self.weight_init_const = np.sqrt(out_n_capsules/(in_d_capsules*in_n_capsules)) 
            # (n,16,m,16)
            self.w = nn.Parameter(self.weight_init_const* \
                                          torch.randn(in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules))
        

        self.dropout = nn.Dropout(dropout)


    def forward(self, current_pose, dots, h_out=1, w_out=1):
        # print("Using Dynamic routing with Bilinear attention")
        # current pose: (b,32,3,3,7,7,16)
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
        if dots is None:
            dots=(torch.zeros(batch_size*h_out*w_out, self.out_n_capsules, self.kernel_size*self.kernel_size*self.in_n_capsules)* (pose_dim ** -0.5)).type_as(current_pose).to(current_pose)
        
        dots = dots.softmax(dim=-2)
        next_pose = current_pose 

        # multiply with N_{L} 
        # next pose: (49b, m, 16) 
        # print(dots.shape, next_pose.shape)
        next_pose = torch.einsum('bij,bje->bie', dots, next_pose)
        
        # Squash the output to make unit length
        # print("Current shape ", next_pose.shape )
        next_pose = squash(next_pose)
        # print("Squashed shape ", next_pose.shape )


        # next_pose = (b,m,7,7,16) --> (49b,m,16)   
        next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
        
        if self.matrix_pose:
            # next_pose = (49b,m,16)  -->  (49b,m,4,4) 
            next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.matrix_pose_dim, self.matrix_pose_dim)
        else:
            next_pose = next_pose.unsqueeze(3)

        # Found final pose of next layer by multiplying X with N_{L}
        # Multiply (49b, m, 4, 4) with (1, m, 4, 4) == (49b, m , 4, 4)
        next_pose = torch.matmul(w_next, next_pose)
        # Reshape: (b, 7, 7, m, 16)
    
        
        #****************       
        if self.matrix_pose:
            # next_pose = (49b,m,16)
            next_pose = next_pose.view(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
        else:
            next_pose = next_pose.squeeze(3)
    

        # Finding scaled alignment scores between updated buckets
        # print("error: ",next_pose.shape, current_pose.shape)
        dr_agreement = torch.einsum('bje,bie->bji', next_pose, current_pose) * (pose_dim ** -0.5) 
        dots= dots + dr_agreement


        # next_pose_candidates = (b,7,7,m,16)
        next_pose = next_pose.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
        
        if self.layer_type == 'conv':
            # next_pose = (b,m,7,7,16)
            next_pose = next_pose.permute([0,3,1,2,4])
        elif self.layer_type == 'FC':
            # next_pose = (b,1,1,m,16) --> (b,1,m,16)
            next_pose = next_pose.squeeze(1)
        
        return dots, next_pose