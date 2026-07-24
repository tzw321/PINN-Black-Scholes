# Physics-Informed Neural Network for Black-Scholes Option Pricing

A Physics-Informed Neural Network (PINN) implementation for solving the **Black-Scholes Partial Differential Equation (PDE)** for European option pricing. Instead of relying solely on labeled training data, the neural network learns mathematical laws of option price.
This project demonstrates how **machine learning** and **financial mathematics** can be combined to solve differential equations without traditional numerical methods such as finite difference or finite element methods.

## Features

- Analytical and Monte Carlo solutions to Black Scholes Model
- Physics-Informed Neural Network implementation using PyTorch
- Solves the Black-Scholes PDE without labeled option prices
- Automatic differentiation for computing derivatives
- Comparison against the analytical Black-Scholes solution
- Visualization of:
  - Training loss
  - Option price surface
  - Prediction error
  - PINN vs Analytical solution
  - Spectral density of Loss Landscape
 
## Packages Used

- PyTorch
- NumPy
- Matplotlib
- SciPy
- PyHessian *

> \* *PyHessian originates from [Yao et al. (2019)](https://arxiv.org/abs/1912.07145). This project uses an adapted version specifically designed for PINNs ([arXiv:2402.01868](https://arxiv.org/pdf/2402.01868)).*
