import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.notebook import tqdm

import tensorly as tl
from tensorly.random import random_tucker
from tensorly.tucker_tensor import tucker_to_tensor


from torch.utils.data import Dataset

#tl.set_backend('pytorch')


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.notebook import tqdm

import tensorly as tl
from tensorly.random import random_tucker
from tensorly.tucker_tensor import tucker_to_tensor


from torch.utils.data import Dataset

#tl.set_backend('pytorch')


class NeuralTensorLayer(torch.nn.Module):
    
    """
    This is the class for the tensor layer
    """
    
    def __init__(self, order, input_dim, output_dim, rank_tucker=-1,
                 initializer=torch.nn.init.xavier_uniform):
        
        super(NeuralTensorLayer, self).__init__()
        
        self.order = order
        self.rank_tucker = rank_tucker
        
        if order > 3 or order < 1:
            raise Exception('Order must be in range [1, 3]')
            
        if rank_tucker != -1 and rank_tucker < 1:
            raise Exception('Tucker rank must be -1 or greater than 0 integer')
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.bias = nn.Parameter(torch.zeros((1, output_dim)), requires_grad=True)
        initializer(self.bias)
        
        self.myparameters = torch.nn.ParameterList([self.bias])
        
        self.order1_tens = self.initialize_n_order_tensor(1, initializer)
        
        if order >= 2:
            self.order2_tens = self.initialize_n_order_tensor(2, initializer)
            
        if order == 3:
            self.order3_tens = self.initialize_n_order_tensor(3, initializer)
        
    # initialize tensor in full or in decomposed form and register it as parameter
    def initialize_n_order_tensor(self, order, initializer):
        
        if self.rank_tucker >= 1:
            
            dim_list = [self.input_dim] * order + [self.output_dim]
            tens_core, factors = random_tucker(dim_list, self.rank_tucker)
            tens_core = nn.Parameter(tens_core, requires_grad=True)
            factors = [nn.Parameter(fact, requires_grad=True) for fact in factors]
            
            self.myparameters.append(tens_core)
            for fact in factors:
                self.myparameters.append(fact)
                
            return (tens_core, factors)
            
        else:
            
            dim_list = [self.input_dim] * order + [self.output_dim]
            var = nn.Parameter(torch.zeros(dim_list), requires_grad=True)
            initializer(var)
            self.myparameters.append(var)
            
            return var

    def compute_result_for_vec(self, core, factor_inp, last_factor): # result dim (1, 1)
        result = core
        for i in range(len(factor_inp)):
            result = tl.tenalg.mode_dot(result, factor_inp[i], i)
        result = result.view(1, -1).mm(torch.transpose(last_factor, 0, 1))
        return result.view(-1)

    def mode_n_dot_accelerated(self, core, factors, input):

        new_factors = [torch.transpose(factors[i], 0, 1).mm(input) for i in range(len(factors) - 1)]

        return torch.stack([
                            self.compute_result_for_vec(core, 
                                                        [new_factors[k][:, i] for k in range(len(factors) - 1)], factors[-1]) 
                            for i in range(input.shape[1])
                            ], dim=0)
        
    def forward(self, X, transposed=False):
        
        #X = torch.Tensor(X)
        
        if self.rank_tucker == -1:
            result = torch.addmm(self.bias, X, self.order1_tens)
        else:
            result = torch.addmm(self.bias, X, tucker_to_tensor(self.order1_tens))
        
        if self.order >= 2:
            
            if self.rank_tucker == -1:      
                acc = tl.tenalg.mode_dot(self.order2_tens, X, 0)
            else:
                acc = tl.tenalg.mode_dot(tucker_to_tensor(self.order2_tens), X, 0)

            acc = tl.tenalg.mode_dot(acc, X, 1)
            result += torch.einsum('iik->ik', acc)
        
        if self.order == 3:
             
            if self.rank_tucker == -1:      
                acc = tl.tenalg.mode_dot(self.order3_tens, X, 0)
            else:
                acc = tl.tenalg.mode_dot(tucker_to_tensor(self.order3_tens), X, 0)
            
            acc = tl.tenalg.mode_dot(acc, X, 1)
            acc = tl.tenalg.mode_dot(acc, X, 2)
            result += torch.einsum('iiik->ik', acc)
        
        return tl.reshape(result, (X.shape[0], self.output_dim))

    def get_orthogonality_loss(self):

        if self.rank_tucker == -1:
            return 0

        loss = 0

        for fact in self.order1_tens[1]:
            loss += torch.sum((tl.dot(fact.T, fact) - torch.eye(fact.shape[1]).cuda()) ** 2)
        
        if self.order >= 2:
            
            for fact in self.order2_tens[1]:
                loss += torch.sum((tl.dot(fact.T, fact) - torch.eye(fact.shape[1]).cuda()) ** 2)
        
        if self.order == 3:
             
            for fact in self.order3_tens[1]:
                loss += torch.sum((tl.dot(fact.T, fact) - torch.eye(fact.shape[1]).cuda()) ** 2)

        return loss


class MOA_set(Dataset):
    
    def __init__(self, fn_X, fn_Y=None, cuda=True):
        self.X = np.load(fn_X)
        self.Y = None if not fn_Y else np.load(fn_Y)
        self.Yn = True if not fn_Y else False
        self.cuda = cuda
        
    def __len__(self):
        return(self.X.shape[0])
    
    def __getitem__(self, ind):
        x = torch.from_numpy(self.X[ind]).type(torch.FloatTensor)
        if self.cuda:
            x = x.cuda()
        if not self.Yn:
            y = torch.from_numpy(self.Y[ind]).type(torch.FloatTensor)
            if self.cuda:
                y = y.cuda()
            return x, y
        else:
            return x




#NN = torch.nn.Sequential(NeuralTensorLayer(2, input_dim, 2, rank_tucker=5), 
 #                        nn.BatchNorm1d(2),
  #                       nn.Softmax(dim=-1))
#loss = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(NN.parameters(), lr=0.001)


# In[ ]:

class MOA_set(Dataset):
    
    def __init__(self, fn_X, fn_Y=None, cuda=True):
        self.X = np.load(fn_X)
        self.Y = None if not fn_Y else np.load(fn_Y)
        self.Yn = True if not fn_Y else False
        self.cuda = cuda
        
    def __len__(self):
        return(self.X.shape[0])
    
    def __getitem__(self, ind):
        x = torch.from_numpy(self.X[ind]).type(torch.FloatTensor)
        if self.cuda:
            x = x.cuda()
        if not self.Yn:
            y = torch.from_numpy(self.Y[ind]).type(torch.FloatTensor)
            if self.cuda:
                y = y.cuda()
            return(x, y)
        else:
            return(x)




