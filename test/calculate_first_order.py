import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
input_dim = 100
hidden_dim = 50
output_dim = 10

import torch

def calculate_gradient_norm(model, x):
    x.requires_grad = True

    y = model(x)
    grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)

    grad_norm = torch.norm(grad[0])
    return grad_norm

model = MLP(input_dim, hidden_dim, output_dim)
input_data = torch.randn(16, input_dim)  # Example input data with batch size 16

gradient_norm = calculate_gradient_norm(model, input_data)
print(gradient_norm)
