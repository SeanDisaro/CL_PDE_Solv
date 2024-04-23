import torch


#define PINN loss function for local solutions:
def D_loc(u, boundary_t, boundary_value , t_grid, f_out, alpha, beta , device = 'cuda'):

  u_out = u(t_grid, boundary_value)
  dudt = torch.autograd.grad(u_out, t_grid, torch.ones_like(t_grid), create_graph = True)[0]


  return alpha * torch.mean( (torch.squeeze(dudt)- f_out)**2 ) + beta * torch.mean(( u(boundary_t, boundary_value) - boundary_value )**2)




#define PINN loss function for global solutions:
def D_glob(u,local_solutions,boundary_t, boundary_rep, boundary_clust,   t_grid, rep_grid, clust_grid , boundary_value , f_out,  alpha, beta ):

  u_out,_ = u(t_grid, rep_grid,clust_grid,  local_solutions)
  dudt = torch.autograd.grad(u_out, t_grid, torch.ones_like(t_grid), create_graph = True, allow_unused=True)[0]


  return alpha * torch.mean( (torch.squeeze(dudt)- f_out)**2 ) + beta * torch.mean(( u(boundary_t, boundary_rep, boundary_clust , local_solutions)[0] - boundary_value )**2)



#define PINN loss function for global solutions (non time dependent boundary):
def D_glob_non_t_dep(u,local_solutions, boundary_rep, boundary_clust,    rep_grid, clust_grid , boundary_value , f_out,  alpha, beta ):

  u_out,_ = u( rep_grid,clust_grid,  local_solutions)
  dudt = torch.autograd.grad(u_out, rep_grid, torch.ones_like(rep_grid), create_graph = True, allow_unused=True)[0]


  return alpha * torch.mean( (torch.squeeze(dudt)- f_out)**2 ) + beta * torch.mean(( u( boundary_rep, boundary_clust , local_solutions)[0] - boundary_value )**2)