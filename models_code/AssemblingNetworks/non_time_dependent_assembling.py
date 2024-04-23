from torch import nn
import torch





#This will be used to put the local solutions together using the local soltions and the clustering
#This only uses boundaryNets for the final solution
class AssemblingNetwork_non_t_dep(nn.Module):
  def __init__(self , n_branch_local_input ,  n_hidden_boundary, n_layers_boundary):
    super().__init__()

    self.activation_func = nn.Tanh

    self.boundaryNetwork_in = nn.Sequential(*[nn.Linear(1,n_hidden_boundary) , self.activation_func()])
    self.boundaryNetwork_hid = nn.Sequential(*[
        nn.Sequential(*[
            nn.Linear(n_hidden_boundary, n_hidden_boundary),
            self.activation_func()
            ])
        for i in range(n_layers_boundary)])

    self.boundaryNetwork_out = nn.Sequential(*[nn.Linear(n_hidden_boundary,n_branch_local_input)])

  def forward(self,  t_representant, i,  local_solutions ):
    #boundary_input = torch.cat([t,i],  dim = 1)

    boundary = self.boundaryNetwork_in(i)
    boundary = self.boundaryNetwork_hid(boundary)
    boundary = self.boundaryNetwork_out(boundary)

    mask_0 = (i == 0)
    mask_1 = (i == 1)


    out = local_solutions[0](t_representant, boundary) * mask_0
    out = out + local_solutions[1](t_representant, boundary) *mask_1



    return out, boundary
