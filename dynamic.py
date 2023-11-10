import torch
import numpy as np

'''
    Dynamic Module:
    This module is for setting up the ODE dynamic. 
'''

torch.set_default_dtype(torch.double)

class odeModule(torch.nn.Module):
    def __init__(self, fOde, theta):
        super().__init__()
        self.f = fOde
        if (not torch.is_tensor(theta)):
            theta = torch.tensor(theta).double()
        self.theta = torch.nn.Parameter(theta)

    def forward(self, x):
        x = x.squeeze()
        if (x.ndimension()==1):
            x = x.reshape(1,-1)
        return (self.f(self.theta, x))
    
class nnModule(torch.nn.Module):
    # Neural Network with Constraint on Output
    def __init__(self, nNodes):
        super().__init__()
        # define layers
        layers = []
        for i in range(len(nNodes)-1):
            layers.append(torch.nn.Linear(nNodes[i],nNodes[i+1],bias=True))
        self.layers = torch.nn.ModuleList(layers).double()
        self.output_dim = nNodes[-1]
        self.output_means = torch.zeros(nNodes[-1])
        self.output_stds = torch.ones(nNodes[-1])

    def update_output_layer(self, means, stds):
        self.output_means = means.detach()
        self.output_stds = stds.detach()

    def forward(self, x):
        if (len(x.size())==1):
            # reshape vector to row length 1 matrix
            x = torch.reshape(x,(1,-1)) 
        for i in range(len(self.layers)-1):
            x = torch.nn.functional.relu(self.layers[i](x))
            # x = torch.tanh(self.layers[i](x))
        x = self.layers[-1](x)
        for i in range(self.output_dim):
            x[:,i] = self.output_means[i] + self.output_stds[i] * x[:,i]
        return (x)

    def reset(self):
        for layer in self.layers:
            layer.reset_parameters()
