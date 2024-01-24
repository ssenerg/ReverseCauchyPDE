import numpy as np
import torch


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

def laplacian(
        inputs: torch.Tensor, 
        gradients: torch.Tensor
    ) -> torch.Tensor:

    laplacians = torch.zeros(inputs.shape)
    # Calculate the second derivatives for each input dimension
    for k in range(inputs.shape[1]):
        # Calculate the gradient of the gradients (second derivatives)
        second_derivatives = torch.autograd.grad(
            gradients[:, k], 
            inputs, 
            grad_outputs=torch.ones_like(gradients[:, k]), 
            create_graph=True
        )[0]
        # Extract the diagonal elements corresponding to the second derivative of 
        # each input
        laplacians[:, k] = second_derivatives[:, k]

    return torch.sum(laplacians ** 2, dim=1, keepdim=True)
