import numpy as np
import plotly.graph_objs as go

# Define the number of points in the grid
grid_points = 300

# Define the grid
x = np.linspace(-0.5, 0.5, grid_points)
y = np.linspace(-0.5, 0.5, grid_points)
grid_x, grid_y = np.meshgrid(x, y)

# Define a function to calculate values (replace this with your actual function)
def value_function(x, y):
    # Example: simple radial function, replace with your own logic
    return np.exp(-(x**2 + y**2))

# Calculate values on the grid
grid_z = value_function(grid_x, grid_y)

# Mask out the area outside the circle
circle_mask = np.sqrt(grid_x**2 + grid_y**2) > 0.5
grid_z[circle_mask] = np.nan

# Create the contour plot
contour = go.Contour(
    x=x,
    y=y,
    z=grid_z,
    colorscale='RdBu',
    ncontours=300,
    showscale=True,
    line=dict(width=0)
)

# Define layout to maintain aspect ratio
layout = go.Layout(
    xaxis=dict(scaleanchor="y", scaleratio=1),
    yaxis=dict(scaleanchor="x", scaleratio=1),
    title="Colored Circle Contour"
)

# Create the figure
fig = go.Figure(data=[contour], layout=layout)

# Show the plot
fig.show()
