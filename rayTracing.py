import numpy as np
import magpylib as magpy
import plotly.graph_objects as go

# Define the magnet
magnet = magpy.magnet.Cuboid(polarization=[0, 0, 1], dimension=[1, 1, 1], position=[0, 0, 0])

# Compute approximate partial derivatives of the magnetic field
def mag_field_derivative(position):
    dx = magnet.getB([position[0] + 0.01, position[1], position[2]]) - magnet.getB([position[0] - 0.01, position[1], position[2]])
    dy = magnet.getB([position[0], position[1] + 0.01, position[2]]) - magnet.getB([position[0], position[1] - 0.01, position[2]])
    dz = magnet.getB([position[0], position[1], position[2] + 0.01]) - magnet.getB([position[0], position[1], position[2] - 0.01])
    return [dx[0] / 0.02, dy[1] / 0.02, dz[2] / 0.02]

# Define the ray class
class Ray:
    def __init__(self, position):
        # Ensure the position is a NumPy array (for vector arithmetic)
        self.position = np.array(position, dtype=float)
        self.trace = [self.position.copy()]
        self.step_size = 0.01
        self.iteration = 0

    def step(self):
        # Take a step along the direction given by the derivative of the magnetic field
        if self.iteration < 1000:
            self.iteration += 1
            deriv = np.array(mag_field_derivative(self.position))
            self.position = self.position + self.step_size * deriv
            self.trace.append(self.position.copy())

# Create a ray starting from an initial position, e.g., [2, 0, 0]
ray_instance = Ray([2, 0, 0])
# Step the ray 1000 times
for i in range(1000):
    ray_instance.step()

# Convert the trace to a NumPy array for easy slicing
trace_array = np.array(ray_instance.trace)

# Create a 3D line plot of the ray trace with Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=trace_array[:, 0],
    y=trace_array[:, 1],
    z=trace_array[:, 2],
    mode='lines',
    line=dict(width=2)
)])

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    title='Ray Trace in Magnetic Field Gradient'
)

fig.show()
