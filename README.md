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

 
## Project Structure

```text
PINN-Black-Scholes/
├── Pinn/
│   ├── funcrtion.py          # functions for training, plotting
│   ├── model.py              # Pytorch PINN model
│   ├── optim.py              # functions to optimize model
│   ├── pinn.ipynb            # main file for pinn with black scholes 
│   └── pyhessian.py          # to evaluate estimate spectral density of loss landscape
│
├── analytical_solution/      # Main source code
│   └── analytical.ipynb      # Analytical Black Scholes solution and monte carlo solution

├── .gitignore                # Git ignore file
├── README.md                 # Project documentation
```
## Packages Used

- PyTorch
- NumPy
- Matplotlib
- SciPy
- PyHessian *
  
---
## References
* **Raissi et al. (2019):** *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving partial differential equations.* [Journal of Computational Physics](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)
* **Louskos (2021):** *Physics-Informed Neural Networks for Pricing Financial Options.* [Dartmouth College Thesis](https://math.dartmouth.edu/theses/undergrad/2021/Louskos-thesis.pdf)
* **Makarov (2023):** *Applications of Physics-Informed Neural Networks in Financial Engineering.* [arXiv:2312.06711](https://arxiv.org/abs/2312.06711)
* **Yao et al., (2019):** *PyHessian for Neural Network Analysis* [arXiv:1912.07145](https://arxiv.org/abs/1912.07145)
* **Pratik et al., (2024):** *Challenges in Training PINNs: A Loss Landscape Perspective* [arXiv:2402.01868](https://arxiv.org/abs/2402.01868)
> \* *PyHessian originates from [Yao et al. (2019)](https://arxiv.org/abs/1912.07145). This project uses an adapted version specifically designed for PINNs ([arXiv:2402.01868](https://arxiv.org/pdf/2402.01868)).*
---

## Future Improvements

- American option pricing
- Asian options
- Heston stochastic volatility model
- Multi-asset Black-Scholes equation
---

