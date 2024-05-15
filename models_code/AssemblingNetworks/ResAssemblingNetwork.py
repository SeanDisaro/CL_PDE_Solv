from torch import nn
import torch



#This will be used to put the local solutions together using the local soltions and the clustering
#This is a variant that uses ResNets for the AssemblingNet
class ResAssemblingNetwork(nn.Module):
  def __init__(self , n_branch_local_input ,n_hidden_assembler, n_layers_assembler,  n_hidden_boundary, n_layers_boundary):
    super().__init__()
    self.activation_func = nn.Tanh

    self.assemblerNetwork_in = nn.Linear(1,n_hidden_assembler)

    self.assemblerNetwork_hid_list = nn.ModuleList([
            nn.Linear(n_hidden_assembler, n_hidden_assembler)
        for _ in range(n_layers_assembler)])

    self.assemblerNetwork_out = nn.Linear(n_hidden_assembler,1)




    self.boundaryNetwork_in = nn.Sequential(*[nn.Linear(2,n_hidden_boundary) , self.activation_func()])

    self.boundaryNetwork_hid = nn.Sequential(*[
        nn.Sequential(*[    nn.Linear(n_hidden_boundary, n_hidden_boundary)  ,    self.activation_func()   ])
        for i in range(n_layers_boundary)] , nn.Linear(n_hidden_boundary, n_hidden_boundary) )

    self.boundaryNetwork_out = nn.Linear(n_hidden_boundary,n_branch_local_input)







  def forward(self, t, t_representant, i,  local_solutions ):

    boundary_input = torch.cat([t,i],  dim = 1)

    boundary = self.boundaryNetwork_in(boundary_input)
    boundary = self.boundaryNetwork_hid(boundary)
    boundary = self.boundaryNetwork_out(boundary)


    mask_0 = (i == 0)
    mask_1 = (i == 1)



    out = local_solutions[0](t_representant, boundary) * mask_0
    out = out + local_solutions[1](t_representant, boundary) *mask_1


    out = self.assemblerNetwork_in(out)

    #ResNet for hid layers:
    for layer in self.assemblerNetwork_hid_list:

      out = nn.functional.tanh(layer(out)) # +out 
    
    out = self.assemblerNetwork_out(out)

    return out


