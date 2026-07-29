import torch
import numpy as np
from function import compute_pinn_loss, eval_model, generate_data
from itertools import product
from model import PINN
import gc


# Train Model with LBFGS
def train_model(data, model, optimizer, epochs, show_progress=False):
    
    loss_history = []
    current_losses = [0.0, 0.0, 0.0, 0.0]

    # Closure function for L-BFGS
    def closure():
        optimizer.zero_grad()
        s, t = data
        loss_components = compute_pinn_loss(s, t, model)
        
        for i in range(4):
            current_losses[i] = loss_components[i].item()
        
        total_loss_tensor = sum(loss_components)
        total_loss_tensor.backward()
        return total_loss_tensor

    if show_progress:
        print(f"Training with {type(optimizer).__name__}...")  
    
    for epoch in range(epochs):
        # Step the optimizer
        if isinstance(optimizer, torch.optim.LBFGS):
            total_loss = optimizer.step(closure)
            total_loss = sum(current_losses)
        else:
            optimizer.zero_grad()
            s, t = data
            loss_components = compute_pinn_loss(s, t, model)

            for i in range(4):
                current_losses[i] = loss_components[i].item()

            total_loss_tensor = sum(loss_components)
            total_loss_tensor.backward()
            optimizer.step()
            total_loss = total_loss_tensor.item()
            
        loss_history.append(total_loss)
        
        if epoch % 500 == 0 and show_progress:
            print(f"Epoch {epoch} | Total Loss: {total_loss:.6f}")
        
    return loss_history



def optim_tune(configs, data):
    results = {}
    for name, cfg in configs.items():
        if name == "Adam+LBFGS":
            model = cfg["model"]
            epochs_Adam = cfg["epochs_Adam"]
            optimizer_Adam = cfg["optimizer_Adam"]
            
            loss_adam = train_model(data, model, optimizer_Adam, epochs_Adam)
            epochs_LBFGS = cfg["epochs_LBFGS"]
            optimizer_LBFGS = cfg["optimizer_LBFGS"]
            loss_LBFGS = train_model(data, model, optimizer_LBFGS, epochs_LBFGS)
            plot_range = epochs_Adam + epochs_LBFGS * optimizer_LBFGS.param_groups[0]['max_iter']
            results[name] = {
                "x": np.concatenate((np.arange(0, epochs_Adam), np.arange(epochs_Adam, plot_range, optimizer_LBFGS.param_groups[0]['max_iter']))),
                "y": loss_adam + loss_LBFGS
            }
        else:
            epochs = cfg["epochs"]
            model = cfg["model"]
            optimizer = cfg["optimizer"]

            loss_y = train_model(data, model, optimizer, epochs)
            
            if name == "LBFGS_wolfe":
                plot_range = epochs * optimizer.param_groups[0]['max_iter']
                results[name] = {
                    "x": np.arange(0, plot_range, optimizer.param_groups[0]['max_iter']),
                    "y": loss_y
                }
            else:
                results[name] = {
                    "x": np.arange(epochs),
                    "y": loss_y
                }
    return results


def gridsearch(grid, training_data, training_params, testing_data, testing_params):
    keys = grid.keys()
    values = grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    models = []
    r, sigma, K, T = testing_params
    print(f"Num of Configs {len(combinations)}")
    for idx, config in enumerate(combinations):
        learning_rate = config['lr']
        layers = config['layers']
        hidden_layers = config['hidden_layers']
        epochs_Adam = config["epochs_Adam"]
        epochs_LBFGS = config["epochs_LBFGS"]
        model = PINN(params=training_params, layers=layers, hidden_layers=hidden_layers)

        # Training
        print(f'Training configs {idx + 1}')
        loss_adam = train_model(training_data, model, torch.optim.Adam(model.parameters(), lr=learning_rate), epochs_Adam)
        loss_LBFGS = train_model(training_data, model, torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, line_search_fn="strong_wolfe"), epochs_LBFGS)

        mse_loss = eval_model(r, sigma, K, T, testing_data, model)
        models.append([model.cpu(), mse_loss])

        del model
        del loss_adam, loss_LBFGS
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return models


        



        