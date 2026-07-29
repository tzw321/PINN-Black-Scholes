import torch.nn as nn

# PINN model
# In features is S, t
class PINN(nn.Module):

    def __init__(self, params: dict, in_features = 2, out_features = 1, layers=3, hidden_layers=64):
        super().__init__()
        self.layers = layers
        self.hidden_layers = hidden_layers
        self.params = params

        layer = [nn.Linear(in_features, hidden_layers), nn.Tanh()]
        for _ in range(layers):
            layer.append(nn.Linear(hidden_layers, hidden_layers))
            layer.append(nn.ReLU())
        layer.append(nn.Linear(hidden_layers, out_features))
        
        self.stack = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.stack(x)

    
