from torch import nn
import torch

#This will be used to compute the local solutions in the representative intervalls [0,2pi], [2pi,4pi] for the clusters A and B
class LocalDeepONet_oneDim(nn.Module):
  def __init__(self, n_branch_input, n_hidden, n_layers):
    super().__init__()
    self.activation_func = nn.Tanh
    self.in_trunk = nn.Sequential(*[nn.Linear(1,n_hidden) , self.activation_func()])
    self.hid_trunk = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh()
            ])
        for i in range(n_layers)], nn.Linear(n_hidden, n_hidden))

    self.in_branch = nn.Sequential(*[nn.Linear(n_branch_input,n_hidden) , self.activation_func()])
    self.hid_branch = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden, n_hidden),
            self.activation_func()
            ], nn.Linear(n_hidden, n_hidden))
        for i in range(n_layers)])

  def forward(self, x_in, boundary):
    x = 0. + x_in
    x = self.in_trunk(x)
    x = self.hid_trunk(x)
    branch = 0. + boundary
    branch = self.in_branch(branch)
    branch = self.hid_branch(branch)
    #use DeepXDE
    out = (x_in - 4*torch.pi)*(x_in - 2*torch.pi )*x_in*torch.sum(x*branch, 1).view(-1,1) + boundary

    return out