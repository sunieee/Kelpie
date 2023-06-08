import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        out = self.fc1(x * y)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def calculate_second_order_derivative(model, x, y):
    x.requires_grad = True
    y.requires_grad = True

    z = model(x, y)
    grad = torch.autograd.grad(outputs=z, inputs=(x, y), grad_outputs=torch.ones_like(z), create_graph=True, retain_graph=True)

    dx_dy_grad = torch.autograd.grad(outputs=grad[0], inputs=y, grad_outputs=torch.ones_like(grad[0]), create_graph=True, retain_graph=True)

    second_order_derivative_norm = torch.norm(dx_dy_grad[0])
    return second_order_derivative_norm

input_dim = 100
hidden_dim = 50
output_dim = 10

model = MLP(input_dim, hidden_dim, output_dim)
x = torch.randn(16, input_dim)  # Example input data for x
y = torch.randn(16, input_dim)  # Example input data for y

second_derivative_norm = calculate_second_order_derivative(model, x, y)
print(second_derivative_norm)

'''
(torch) sy@actvis:~/2023/Kelpie-copy/test$ python calculate_second_order.py 
tensor(1.2365, grad_fn=<LinalgVectorNormBackward0>)
(torch) sy@actvis:~/2023/Kelpie-copy/test$ python calculate_second_order.py 
tensor(1.5067, grad_fn=<LinalgVectorNormBackward0>)
'''