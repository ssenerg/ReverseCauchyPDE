import numpy as np

def finite_difference_derivative(data, axis, h):
    """
    Calculate the derivative of a data array along a specified axis using central differences.
    
    :param data: n-dimensional array of data points.
    :param axis: axis along which to take the derivative.
    :param h: array of sample spacings along the axis.
    :return: derivative of the data along the given axis.
    """
    # Ensure h has the correct size
    if data.shape[axis] != len(h) + 1:
        raise ValueError("Spacing array h must have one less element than the size of data along the specified axis.")

    derivative = np.zeros_like(data)
    
    # Central differences for interior points
    for i in range(1, len(h)):
        slice_left = [slice(None)] * data.ndim
        slice_right = [slice(None)] * data.ndim
        slice_center = [slice(None)] * data.ndim
        
        slice_left[axis] = i - 1
        slice_right[axis] = i + 1
        slice_center[axis] = i
        
        derivative[tuple(slice_center)] = (data[tuple(slice_right)] - data[tuple(slice_left)]) / (h[i] + h[i-1])
    
    # Forward difference for the first boundary
    slice_begin = [slice(None)] * data.ndim
    slice_second = [slice(None)] * data.ndim
    
    slice_begin[axis] = 0
    slice_second[axis] = 1
    
    derivative[tuple(slice_begin)] = (data[tuple(slice_second)] - data[tuple(slice_begin)]) / h[0]
    
    # Backward difference for the last boundary
    slice_end = [slice(None)] * data.ndim
    slice_penultimate = [slice(None)] * data.ndim
    
    slice_end[axis] = -1
    slice_penultimate[axis] = -2
    
    derivative[tuple(slice_end)] = (data[tuple(slice_end)] - data[tuple(slice_penultimate)]) / h[-1]
    
    return derivative


def laplacian(data, spacings):
    """
    Calculate the Laplacian of a data array given the spacings for each axis.
    
    :param data: n-dimensional array of data points.
    :param spacings: list of arrays representing the spacings for each axis.
    :return: Laplacian of the data.
    """
    laplacian = np.zeros_like(data)
    for axis, h in enumerate(spacings):
        second_derivative = finite_difference_derivative(
            finite_difference_derivative(data, axis, h), axis, h)
        laplacian += second_derivative
    
    return laplacian

# Example usage:
# Suppose we have a 4D data array `u` with dimensions corresponding to 2 spatial dimensions, time, and the function value.
# We also have arrays `x1`, `x2`, and `t` representing the non-uniform spacing in each dimension.

# Create some example data (replace this with your actual data)

surface_dimensions = 2
t_samples = 50
d_samples = 100
e_samples = 20


t = np.random.rand(t_samples)
u_domain_t = np.random.rand(d_samples, t_samples)
x_domain = np.random.rand(d_samples, surface_dimensions)
u_edge_t = np.random.rand(e_samples, t_samples)
x_edge = np.random.rand(e_samples, surface_dimensions)


import tensorflow as tf
import numpy as np

# Constants
surface_dimensions = 2
t_samples = 50
d_samples = 100
e_samples = 20

# Sample data
t = np.random.rand(t_samples)
x_domain = np.random.rand(d_samples, surface_dimensions)
x_edge = np.random.rand(e_samples, surface_dimensions)

# Define LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(None, surface_dimensions + 1), return_sequences=True),
    tf.keras.layers.Dense(1)
])

# Custom loss function to compute derivatives
def custom_loss(y_true, y_pred):
    # Compute the gradient of u with respect to t and x
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        for dim in range(surface_dimensions):
            tape.watch(x_domain[:, dim])
        u_pred = model(tf.concat([x_domain, tf.reshape(t, (1, -1))], axis=1))
    
    du_dt = tape.gradient(u_pred, t)
    gradients_x = [tape.gradient(u_pred, x_domain[:, dim]) for dim in range(surface_dimensions)]
    
    # Compute the mean squared error for the gradient loss
    loss = tf.reduce_mean(tf.square(du_dt - y_true))  # Placeholder for actual gradient comparison
    for grad in gradients_x:
        loss += tf.reduce_mean(tf.square(grad - y_true))  # Placeholder for actual gradient comparison
    
    return loss

# Compile the model
model.compile(optimizer='adam', loss=custom_loss)

# Prepare the input data
# Note: You need to reshape or process your input data to match the expected input shape for the LSTM.
input_data = np.concatenate((x_domain, np.tile(t, (d_samples, 1))), axis=1)
input_data = input_data.reshape((d_samples, t_samples, surface_dimensions + 1))

# Dummy target data (you'll need to replace this with your actual target data)
target_data = np.zeros((d_samples, t_samples, 1))

# Train the model
model.fit(input_data, target_data, epochs=10)
