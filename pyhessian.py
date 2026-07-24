'''
PyHessian original paper: PyHessian: Neural Networks Through the Lens of the Hessian (https://arxiv.org/abs/1912.07145)
PyHessian original implementation: https://github.com/amirgholami/PyHessian

The code below are from another author PyHessian that is adapted for PINN

Original paper: Challenges in Training PINNs: A Loss Landscape Perspective (https://arxiv.org/abs/2402.01868)
Original authors implementation: https://github.com/pratikrathore8/opt_for_pinns

Some code are changed to align with main.ipynb file. 
'''

import torch
import numpy as np
from matplotlib import pyplot as plt

### utility functions ###

"""
Compute the inner product of two lists of tensors xs, ys.
"""
def group_product(xs, ys):
  return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

"""
Compute the updated list of tensors (params) in the corresponding list of direction tensors (update) with given step size (alpha). 
"""
def group_add(params, update, alpha=1):
  for i, p in enumerate(params):
    p.data.add_(update[i], alpha=alpha)
  return params

"""
Normalize a vector (represented as a list of tensors). . 
"""
def normalization(v):
  s = group_product(v, v)
  s = s ** 0.5
  s = s.cpu().item()
  v = [vi / (s + 1e-6) for vi in v]
  return v

"""
Orthonormalize vector w to a list of vectors (v_list). 
"""
def orthonormalization(w, v_list):
  for v in v_list:
    w = group_add(w, v, alpha=-group_product(w, v))
  return normalization(w)

"""
Compute the HVP where gradsH is the gradient at the current point, params is the corresponding variables, and v is the vector. 
"""
def hessian_vector_product(gradsH, params, v):
  hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=True)
  return hv

### hessian class ###

class hessian():
  """
  Class for computing spectral density of Hessian. 

  - model: instance of PINN
  - loss_func: loss function
  - data: tuple of spatial and temporal inputs
  - device: string indicating which CUDA device to use (where both model and data reside)
  """
  def __init__(self, model, loss_func, data, device='cuda'):
    self.model = model.eval()
    self.loss_func = loss_func
    self.s, self.t = data
    self.device = device

    self.model.zero_grad()
    loss = sum(self.loss_func(self.s, self.t, self.model))

    grad_tuple = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        
    self.params = [param for param in self.model.parameters() if param.requires_grad]
    self.gradsH = [gradient if gradient is not None else 0.0 for gradient in grad_tuple]

  """
  Function for performing spectral density computation. 

  INPUT: 
  - num_iter: number of iterations for Lanczos
  - num_run: number of runs
  OUTPUT: 
  - eigen_list_full: list eigenvalues for each run
  - weight_list_full: list of corresponding densities for each run
  """
  def density(self, num_iter=100, num_run=1):
    eigen_list_full = []
    weight_list_full = []

    for k in range(num_run):
      # generate Rademacher random variables
      v = [2 * torch.randint_like(p, high=2, device=self.device) - 1 for p in self.params]
      v = normalization(v)

      # Lanczos initlization
      v_list = [v]
      w_list = []
      alpha_list = []
      beta_list = []

      # run Lanczos
      for i in range(num_iter):
        self.model.zero_grad()
        w_prime = [torch.zeros(p.size()).to(self.device) for p in self.params]
        if i == 0:
          w_prime = hessian_vector_product(self.gradsH, self.params, v)
          alpha = group_product(w_prime, v)
          alpha_list.append(alpha.cpu().item())
          w = group_add(w_prime, v, alpha=-alpha)
          w_list.append(w)
        else:
          beta = torch.sqrt(group_product(w, w))
          beta_list.append(beta.cpu().item())
          if beta_list[-1] != 0.:
            v = orthonormalization(w, v_list)
            v_list.append(v)
          else:
            w = [torch.randn(p.size()).to(self.device) for p in self.params]
            v = orthonormalization(w, v_list)
            v_list.append(v)
          w_prime = hessian_vector_product(self.gradsH, self.params, v)
          alpha = group_product(w_prime, v)
          alpha_list.append(alpha.cpu().item())
          w_tmp = group_add(w_prime, v, alpha=-alpha)
          w = group_add(w_tmp, v_list[-2], alpha=-beta)

      # piece together tridiagonal matrix
      T = torch.zeros(num_iter, num_iter).to(self.device)
      for i in range(len(alpha_list)):
        T[i, i] = alpha_list[i]
        if i < len(alpha_list) - 1:
          T[i + 1, i] = beta_list[i]
          T[i, i + 1] = beta_list[i]

      eigenvalues, eigenvectors = torch.linalg.eig(T)

      eigen_list = eigenvalues.real
      weight_list = torch.pow(eigenvectors[0,:], 2)
      eigen_list_full.append(list(eigen_list.cpu().numpy()))
      weight_list_full.append(list(weight_list.cpu().numpy()))

    return eigen_list_full, weight_list_full

  """
  Function for finding eigenvalues and eigenvectors using power iteration. 

  INPUT: 
  - max_num_iter: maximum number of iterations for each eigenvalue
  - top_n: number of top n eigenvalues to discover
  OUTPUT: 
  - eigenvalues: eigenvalues
  - eigenvectors: corresponding eigenvectors
  - iter_used: number of iterations until convergence for each eigenvalue
  """
  def eigenvalues(self, max_num_iter=100, tol=1e-3, top_n=1):
    assert top_n >= 1

    device = self.device

    eigenvalues = []
    eigenvectors = []
    iter_used = np.zeros(top_n)

    computed_dim = 0

    while computed_dim < top_n:
        eigenvalue = None
        v = [torch.randn(p.size()).to(device) for p in self.params]  # generate random vector
        v = normalization(v)  # normalize the vector

        for i in range(max_num_iter):
            iter_used[computed_dim] += 1
            v = orthonormalization(v, eigenvectors)
            self.model.zero_grad()

            Hv = hessian_vector_product(self.gradsH, self.params, v)
            tmp_eigenvalue = group_product(Hv, v).cpu().item()

            v = normalization(Hv)

            if eigenvalue == None:
                eigenvalue = tmp_eigenvalue
            else:
                if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                    break
                else:
                    eigenvalue = tmp_eigenvalue
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        computed_dim += 1

    return eigenvalues, eigenvectors, iter_used


def generate_density(eigenvalues, weights, num_grid_points=int(1e5), sigma_squared=1e-5, boundary_margin=1e-3):

  def gaussian(x, x0, variance):
    return np.exp(-(x0 - x)**2 / (2.0 * variance)) / np.sqrt(2 * np.pi * variance)

  eigenvalues = np.array(eigenvalues).real
  weights = np.array(weights).real

  num_runs, num_eigvals = eigenvalues.shape

  lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + boundary_margin
  lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - boundary_margin

  grid_points = np.linspace(lambda_min, lambda_max, num=num_grid_points)
  grid_length = (lambda_max - lambda_min) / (num_grid_points - 1)

  sigma = sigma_squared * max(1, (lambda_max - lambda_min))

  # compute density
  densities = np.zeros((num_runs, num_grid_points), dtype=eigenvalues.dtype)
  for i in range(num_runs):
    for j in range(num_grid_points):
      x = grid_points[j]
      convolutions = gaussian(eigenvalues[i,:], x, sigma)
      densities[i,j] = np.sum(convolutions * weights[i,:])

  # average across runs
  densities = np.mean(densities, axis=0)
  densities = densities / (np.sum(densities) * grid_length)

  return densities, grid_points

def get_esd_plot(eigenvalues, weights):
    density, grids = generate_density(eigenvalues, weights)
    return density, grids