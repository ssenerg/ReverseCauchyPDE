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



def laplacian(model, inputs):
    model.eval()
    inputs.requires_grad_(True)

    # First forward pass
    outputs = model(inputs).squeeze()

    first_derivatives = torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    # We now compute the gradients of each element of first_derivatives w.r.t. inputs
    second_derivatives = torch.autograd.grad(
        first_derivatives, inputs, grad_outputs=torch.ones_like(first_derivatives),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

    # Assuming a scalar output, the Laplacian is the sum of the second derivatives
    laplacian = second_derivatives.sum(dim=tuple(range(1, second_derivatives.ndimension())))
    return laplacian

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

criterion = nn.MSELoss()
pde = ReverseChauchyPDE(
    f_function=F_EXPR,
    g_function=G_EXPR,
    h_function=H_EXPR,
    x_symbols=Xs_symbol,
    time_symbol=t_symbol,
    criterion=criterion
)


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
# batch_size = 256
batch_size = domain_input.shape[0] + edge_input.shape[0]
num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    for i in range(0, combined_data.size(0), batch_size):
        # Get the batch
        inputs = combined_data[i:i+batch_size]
        outputs = model(inputs)

        # Calculate the derivatives for each feature in the input
        laplacian = laplacian(model, inputs)
        gradient = derivative(model, inputs)

        tr_loss = criterion(laplacian + gradient[:, -1], torch.zeros_like(laplacian))
        f_loss = pde.loss('f', outputs, gradient)
        g_loss = pde.loss('g', outputs, gradient)
        h_loss = pde.loss('h', outputs, gradient)

        # Combine losses
        combined_loss = tr_loss + f_loss + g_loss + h_loss

        # Backward pass
        combined_loss.backward()
        optimizer.step()

        # Update total losses
        total_tr_loss += tr_loss.item()
        total_f_loss += f_loss.item()
        total_g_loss += g_loss.item()
        total_h_loss += h_loss.item()
    
    # Print average losses after each epoch
    print('Epoch [{}/{}], TR Loss: {:.4f}, F Loss: {:.4f}, G Loss: {:.4f}, H Loss: {:.4f}'
          .format(epoch+1, num_epochs, total_tr_loss/(i+1), total_f_loss/(i+1), total_g_loss/(i+1), total_h_loss/(i+1)))
