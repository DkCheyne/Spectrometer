import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, position, velocity, charge, mass, dt):
        self.position = np.array(position, dtype=float)  # (x, y, z)
        self.prev_position = self.position - np.array(velocity, dtype=float) * dt  # Estimate x_prev
        self.charge = charge  # Charge (C)
        self.mass = mass  # Mass (kg)
        self.dt = dt  # Time step (s)

    def compute_lorentz_force(self, B_field):
        """Computes acceleration from Lorentz Force: F = q (v × B) / m"""
        velocity = (self.position - self.prev_position) / self.dt  # Estimate velocity
        force = self.charge * np.cross(velocity, B_field)  # F = q (v × B)
        acceleration = force / self.mass  # a = F / m
        return acceleration

    def verlet_update(self, B_field):
        """Verlet integration step for updating position based on acceleration"""
        acceleration = self.compute_lorentz_force(B_field)
        new_position = 2 * self.position - self.prev_position + acceleration * self.dt**2
        self.prev_position = self.position  # Store old position
        self.position = new_position  # Update to new position

# Simulation Parameters
dt = 1e-9  # Timestep in seconds
steps = 1000  # Number of simulation steps

# Define a particle with initial position, velocity, charge, and mass
particle = Particle(position=(0, 0, 0), velocity=(1e6, 1e6, 0), charge=1.6e-19, mass=9.11e-31, dt=dt)

# Uniform magnetic field (e.g., along the z-axis)
B_field = np.array([0, 0, 1])  # Tesla

# Store trajectory
trajectory = []

# Run simulation
for _ in range(steps):
    trajectory.append(particle.position.copy())
    particle.verlet_update(B_field)

# Convert trajectory to array for plotting
trajectory = np.array(trajectory)

# Plot the trajectory in 2D
plt.figure(figsize=(6, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Particle Path")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Particle Motion with Verlet Integration")
plt.legend()
plt.grid()
plt.show()
