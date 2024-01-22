import numpy as np
import torch


gpu = torch.device("mps") if torch.backends.mps.is_available() else \
    torch.device("cpu")
gpu = torch.device("cpu")
cpu = torch.device("cpu")


def laplacian(model, inputs):
    return


def derivative(model, inputs):
    model.eval()

    inputs.requires_grad_(True)

    # Forward pass to get the outputs
    outputs = model(inputs).squeeze()

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

