import numpy as np

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

# Example datasets
a = np.random.rand(4, 2)  # Replace with your actual dataset (500, n), assuming n=2 here
b = np.linspace(0, 5, 5)  # Replace with your actual dataset (250,

print(a)
print()
print(b)
print()
print(prepare_data(b, a))        # Should output the Cartesian product of a and b