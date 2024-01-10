from ANNPDE.PDE.shapes import (
    ElipseShape, 
    CircleShape, 
    LineShape
)
from ANNPDE.PDE import ReverseChauchyPDE
import plotly.graph_objs as go
from random import randint
from torch import nn
import numpy as np
import torch



SEED = randint(1, 1000000)
print('Seed:', SEED)

F_EXPR = 'E ** x1 * sin(x2) * cos(t)'
G_EXPR = ['E ** x1 * sin(x2)', 'E ** x1 * cos(x2)']
H_EXPR = 'E ** x1 * sin(x2)'
Xs_symbol, t_symbol = ['x1', 'x2'], 't'

T_SAMPLE = 256
E_SAMPLE = 128
D_SAMPLE = 2048
CENTER = np.array([0, 0])
RADIUS = 10

time = LineShape(
    seed=SEED,
    n=T_SAMPLE,
    start_point=0,
    end_point=np.pi/2,
    cross_sample_generate=1,
    even_sample=True
)

time_sample = time.get()

shape = CircleShape(
    seed=SEED,
    edge_n=E_SAMPLE,
    domain_n=D_SAMPLE,
    center=CENTER,
    radius=RADIUS,
    cross_sample_generate=1,
    even_sample=True
)

edge_sample, domain_sample = shape.get()

pde = ReverseChauchyPDE(
    f_function=F_EXPR,
    g_function=G_EXPR,
    h_function=H_EXPR,
    x_symbols=Xs_symbol,
    domain_sample=domain_sample,
    edge_sample=edge_sample,
    time_sample=time_sample,
    time_symbol=t_symbol
)

# ----------- 

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size=1):
        super(CustomLSTM, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layer_sizes)):
            if i == 0:
                self.hidden_layers.append(
                    nn.LSTM(input_size, hidden_layer_sizes[i], batch_first=True)
                )
            else:
                self.hidden_layers.append(
                    nn.LSTM(
                        hidden_layer_sizes[i-1], 
                        hidden_layer_sizes[i]
                    )
                )
        
        self.linear = nn.Linear(hidden_layer_sizes[-1], output_size)

    def forward(self, x):
        for lstm in self.hidden_layers:
            x, _ = lstm(x)
        x = self.linear(x)  # Take the last time step's output
        return x


def derivative(model, inputs):
    # Ensure the model is in evaluation mode if it contains any dropout or batchnorm layers
    model.eval()

    # Make sure inputs have requires_grad set to True so we can get gradients
    inputs.requires_grad_(True)

    # Forward pass to get the outputs
    outputs = model(inputs)

    # We assume outputs are of shape [batch_size, 1] for a single neuron
    # If your output has a different shape, adjust accordingly
    outputs = outputs.squeeze()

    # Initialize a tensor to hold gradients
    gradients = torch.zeros_like(inputs)

    # Calculate gradients for each sample in the batch
    for i in range(outputs.size(0)):
        # Zero out previous gradients
        model.zero_grad()

        # Select the output of the current sample and backpropagate
        output_i = outputs[i]
        output_i.backward(retain_graph=True)  # retain_graph=True allows multiple backward passes

        # Extract the gradients for the current input
        gradients[i] = inputs.grad[i].detach()  # Detach to prevent further graph operations

        # Zero the gradients on the inputs after extracting them
        inputs.grad.data.zero_()

    # Return the gradients
    return gradients


# Function to prepare the data for LSTM input
def prepare_data(time_sample, surface_sample):
    time_sample_ = time_sample.reshape(-1, 1)
    surface_sample_ = np.repeat(
        surface_sample[np.newaxis, :, :], repeats=time_sample.shape[0], axis=0
    )
    time_sample_ = np.repeat(
        time_sample_[:, np.newaxis, :], repeats=surface_sample.shape[0], axis=1
    )
    cartesian_product = np.concatenate((surface_sample_, time_sample_), axis=2)
    return cartesian_product.reshape(-1, 3)

# Example usage
n = 2  # Dimension of the surface
input_size = n + 1  # Input is (x1, x2, t)
hidden_layer_sizes = [10, 20, 50, 90, 150, 100, 50]  # Customize your hidden layer sizes
output_size = 1  # Output is a single number

# Initialize the model
model = CustomLSTM(input_size, hidden_layer_sizes, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Convert numpy arrays to PyTorch tensors and reshape time_sample for concatenation
domain_input = torch.from_numpy(prepare_data(time_sample, domain_sample)).float()
edge_input = torch.from_numpy(prepare_data(time_sample, edge_sample)).float()

# Assume some target values for demonstration (you should replace these with your actual targets)
domain_targets = torch.randn(domain_input.size(0), 1)
edge_targets = torch.randn(edge_input.size(0), 1)

# Concatenate domain and edge data and targets
combined_data = torch.cat((domain_input, edge_input), dim=0)
combined_targets = torch.cat((domain_targets, edge_targets), dim=0)

# Shuffle the data
perm = torch.randperm(combined_data.size(0))
combined_data = combined_data[perm]

# Define batch size
batch_size = 250_000
# batch_size = domain_input.shape[0] + edge_input.shape[0]
num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    for i in range(0, combined_data.size(0), batch_size):
        # Get the batch
        inputs = combined_data[i:i+batch_size]
        outputs = model(inputs)

        # Calculate the derivatives for each feature in the input
        gradient = derivative(model, inputs)
        print(gradient.shape)

        # Calculate loss for domain data
        # domain_loss = criterion(outputs[:domain_input.shape[0]], domain_targets)
        # edge_loss = criterion(outputs[domain_input.shape[0]:], g_values)

        # Combine losses
        loss = domain_loss + edge_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')