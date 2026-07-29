import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import norm

# Solving Black Scholes Analytically
def BS(r, sigma, S0, K, T):
    d1 = (torch.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    N_d1 = norm.cdf(d1, 0, 1)
    N_d2 = norm.cdf(d2, 0, 1)
    C = S0 * N_d1 - K * torch.exp(-r * T)* N_d2
    return C

# Solving Black Scholes Using Monte Carlo
def BS_MC(r, sigma, S0, K, T, num_samples):
    Z = np.random.standard_normal(num_samples)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

def collocation_points(S_max, T, n_pde=5000, n_boundary=500):
     # 1. Interior Domain Points (Randomly sampled PDE points)
    t_pde = (torch.rand(n_pde) * T)
    s_pde = (torch.rand(n_pde) * S_max)

    # 2. Lower Boundary (S = 0)
    t_low = torch.rand(n_boundary) * T
    s_low = torch.zeros(n_boundary)

    # 3. Upper Boundary (S = S_max)
    t_high = torch.rand(n_boundary) * T
    s_high = torch.ones(n_boundary) * S_max

    # 4. Terminal Payoff Boundary (t = T)
    t_term = torch.ones(n_boundary) * T
    s_term = torch.rand(n_boundary) * S_max

    s = [s_pde, s_low, s_high, s_term]
    t = [t_pde, t_low, t_high, t_term]

    s[0].requires_grad_(True)
    t[0].requires_grad_(True)
    
    return s, t

def plot_collocation(S_max, T):
    s, t = collocation_points(S_max, T)
    s_pde, s_low, s_high, s_term = s
    t_pde, t_low, t_high, t_term = t

    plt.figure(figsize=(10, 6))
    
    # 1. Interior Domain Points (Randomly sampled PDE points)
    plt.scatter(t_pde.detach().numpy(), s_pde.detach().numpy(), 
                color='royalblue', alpha=0.4, s=10, label='Interior PDE Points')

    # 2. Lower Boundary (S = 0)
    plt.scatter(t_low.numpy(), s_low.numpy(), 
                color='crimson', marker='o', s=15, label='Lower Bound: $C(0, t) = 0$')

    # 3. Upper Boundary (S = S_max)
    plt.scatter(t_high.numpy(), s_high.numpy(), 
                color='darkorange', marker='o', s=15, label='Upper Bound: $C(S_{max}, t)$')

    # 4. Terminal Payoff Boundary (t = T)
    plt.scatter(t_term.numpy(), s_term.numpy(), 
                color='forestgreen', marker='x', s=15, linewidths=2, label='Terminal Payoff: $t = T$')

    plt.title('PINN Collocation Points & Boundary Domains', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Time to Maturity ($t$)', fontsize=12)
    plt.ylabel('Stock Price ($S$)', fontsize=12)
    
    # Clean grid and neat layout margins
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-0.02, T + 0.05)
    plt.ylim(-5, S_max + 10)
    
    # Move legend outside or clear from points
    plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='gainsboro', fontsize=10)
    plt.tight_layout()
    plt.show()


def compute_pinn_loss(s, t, model):
    params = model.params
    r = params['r']          
    sigma = params['sigma']   
    K = params['K']           
    T = params['T']           
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    s_pde, s_low, s_high, s_term = [x.to(device).unsqueeze(1) for x in s]
    t_pde, t_low, t_high, t_term = [x.to(device).unsqueeze(1) for x in t]
    
    # Lower Boundary Loss: V(0, t) = 0
    pred_low = model(torch.cat((s_low, t_low), dim=1))
    bc1_loss = F.mse_loss(pred_low, torch.zeros_like(pred_low))

    # Upper Boundary Loss: V(S_max, t) ≈ S_max - K * e^(-r*(T-t))
    pred_high = model(torch.cat((s_high, t_high), dim=1))
    target_high = s_high - K * torch.exp(-r * (T - t_high))
    bc2_loss = F.mse_loss(pred_high, target_high)

    # Terminal Payoff Loss: V(S, T) = max(S - K, 0)
    pred_term = model(torch.cat((s_term, t_term), dim=1))
    target_term = torch.clamp(s_term - K, min=0.0)
    terminal_loss = F.mse_loss(pred_term, target_term)

    # PDE Loss (Black-Scholes Residual)
    pred_pde = model(torch.cat((s_pde, t_pde), dim=1))
    
    dcs = torch.autograd.grad(pred_pde, s_pde, grad_outputs=torch.ones_like(pred_pde), create_graph=True)[0]
    d2cs = torch.autograd.grad(dcs, s_pde, grad_outputs=torch.ones_like(dcs), create_graph=True)[0]
    dct = torch.autograd.grad(pred_pde, t_pde, grad_outputs=torch.ones_like(pred_pde), create_graph=True)[0]

    pde_residual = r * s_pde * dcs + 1/2 * sigma ** 2 * s_pde ** 2 * d2cs - r * pred_pde + dct
    pde_loss = F.mse_loss(pde_residual, torch.zeros_like(pde_residual)) 
    
    return [bc1_loss, bc2_loss, terminal_loss, pde_loss]


# pred function with PINN model
def pred_model(grid, model):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        y_pinn = model(grid.to(device)) 
    return y_pinn.squeeze().detach().cpu()


def L2_Loss(pred, target):
    return (pred - target) ** 2


# Evaluate model with analytical solution and calcualtes the mse error
def eval_model(r, sigma, K, T, grid, model, plotting=False):
    s, t = grid
    eval_pts = torch.stack([s.reshape(-1), t.reshape(-1)], dim=-1)
    mse_loss = 0
    with torch.no_grad():
        bs_analytical = BS(r, sigma, s, K, T - t)
        bs_pinn = pred_model(eval_pts, model).reshape(s.shape[0], t.shape[1])
        mse_loss = L2_Loss(bs_pinn, bs_analytical)
    
    if plotting:
        plot_error_heatmap(grid, mse_loss)

    mse_loss = mse_loss.flatten()

    return torch.sum(mse_loss) / len(mse_loss)

def plot_error_heatmap(grid, error_grid):

    plt.figure(figsize=(9, 6))
    x, y = grid
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    z = error_grid.cpu().numpy()
    mesh = plt.pcolormesh(x, y, z, shading="auto", cmap="viridis", vmin=0)
    plt.title("PINN Prediction MSE Error", fontsize=13, fontweight="bold")
    cbar = plt.colorbar(mesh)
    cbar.set_label(
        "MSE Error",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel("Stock Price ($S$)", fontsize=11, fontweight="bold")
    plt.ylabel("Time ($t$)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_slices(r, sigma, K, T, model, n_slice=3):
    fig, axs = plt.subplots(ncols=n_slice, figsize=(15,5))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S_vals = torch.linspace(5, 150, 50)
    time = torch.linspace(0, T, n_slice)
    for idx, t in enumerate(time):
        ax = axs[idx]
        # Analytical
        bs_analytical = [BS(r, sigma, s, K, T - t) for s in S_vals]
        # PINN
        t_vals = t.repeat(len(S_vals))
        grid = torch.stack((S_vals, t_vals), dim=-1).to(device)
        bs_pinn = pred_model(grid, model)
        
        ax.plot(S_vals, bs_analytical, label='Analytical Black-Scholes', linewidth=2)
        ax.plot(S_vals, bs_pinn, label='PINN Black-Scholes')
        ax.set_title(f'Option Price at $t = {t:.2f}$', fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)

    axs[0].set_ylabel('Call Option', fontsize=11, fontweight='bold')
    for col in range(n_slice):
        axs[col].set_xlabel('Stock Price ($S_0$)', fontsize=11, fontweight='bold')


def generate_data(randomness=False, seed=None, t_num=100, s_num=100):
    time = torch.linspace(0, 0.95, t_num) 
    S_vals = torch.linspace(5, 150, s_num) 
    eval_pts = torch.meshgrid(S_vals, time, indexing='xy')

    if randomness:
        random.seed(seed)
        r = random.uniform(0, 0.1)
        sigma = random.uniform(0.1, 0.5)
        K = random.uniform(50, 150)
        T = random.uniform(1, 5)
        return r, sigma, K, T, eval_pts

    return eval_pts
    

