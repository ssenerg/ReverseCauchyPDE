import torch
import numpy as np


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

