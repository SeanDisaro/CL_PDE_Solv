import numpy as np
import torch 
from torch import nn


#Define clustering funciton
def clustering(t):
  n = t // (4* torch.pi)
  t_proj = t - n*4* torch.pi
  if t_proj < 2* torch.pi:
    #then we are in cluster A
    return (0,t_proj)
  else:
    #then we are in cluster B
    return (1, t_proj)
  
#Define data function
def f(t):
  clustering_t = clustering(t)
  if clustering_t[0] == 0:
    return 2* torch.sin(t)
  else:
    return torch.sin(2* t)
  

#exact solution to IVP
def exact_sol(t):
  clustering_t = clustering(t)
  if clustering_t[0] == 0:
    return -2* torch.cos(t)+2
  else:
    return - torch.cos(2* t) /2 +1/2
  

def exact_sol_tensor(my_tensor, b):
  result = torch.clone(my_tensor)
  for i in range(my_tensor.shape[0]):
    result[i] = exact_sol(my_tensor[i])
  return result