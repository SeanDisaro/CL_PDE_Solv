import numpy as np
import torch 
from torch import nn


#Define clustering funciton
def clustering(t):
  n = t // (2* torch.pi)
  t_proj = t - n*2* torch.pi
  return 0, t_proj

#clustering function for grid:
def clustering_grid( tgrid ):
  n = tgrid.shape[0]
  result = np.empty((n,2))
  for i in range(n):
    result[i] = clustering(tgrid[i])

  return result

#tgrid = torch.rand(1000)
#print(clustering_grid( tgrid ))
  
#Define data function
def f(t):
  return t * torch.sin(t)


#exact solution to IVP
def exact_sol(t):
  return torch.sin(t) - t* torch.cos(t)


def exact_sol_tensor(my_tensor, b):
  result = torch.clone(my_tensor)
  for i in range(my_tensor.shape[0]):
    result[i] = exact_sol(my_tensor[i])
  return result