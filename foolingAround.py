import magpylib as magpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define two parallel plate magnets as thin cuboids.
plate1 = magpy.magnet.Cuboid(
    magnetization=(1, 0, 0),  # magnetized in +x direction
    dimension=(0.1, 2, 1),    # thin plate: thickness 0.1 in x, 2 in y, 1 in z
    position=(-1, 0, 0)       # positioned at x = -1
)

plate2 = magpy.magnet.Cuboid(
    magnetization=(-1, 0, 0),  # magnetized in -x direction
    dimension=(0.1, 2, 1),
    position=(1, 0, 0)         # positioned at x = 1
)

def magnetic_field(position):
    """
    Given a 3D position, compute the total magnetic field from both plates.
    Ensures proper dimensionality.
    """
    pos = np.atleast_2d(position)
    B_total = plate1.getB(pos) + plate2.getB(pos)
    return np.squeeze(B_total)  # ensures output is shape (3,)

def particle_deriv(t, y, q, m):
    """
    Computes the derivative for the ODE system.
    y contains [x, y, z, vx, vy, vz].
    Returns dy/dt = [vx, vy, vz, ax, ay, az],
    where acceleration a = (q/m)*(v x B).
    """
    position = y[:3]
    velocity = y[3:]
    B = magnetic_field(position)
    acceleration = (q / m) * np.cross(velocity, B)
    return np.concatenate([velocity, acceleration])

# Parameters for the charged particle.
q = 50.0  # increased charge for stronger effect
m = 1.0   # mass (arbitrary units)

# Initial conditions: starting at (-2, 0, 0) with velocity purely in the y direction.
y0 = np.array([-0, 0, 0, 0, 1, 0])
# You can try modifying the initial velocity, for example:
# y0 = np.array([-0.5, -2, 0, 0, 1, 0])
# to ensure the particle starts between the plates.

# Time span for the simulation.
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve the ODE system using solve_ivp.
sol = solve_ivp(particle_deriv, t_span, y0, t_eval=t_eval, args=(q, m), rtol=1e-8)

# Extract the trajectory.
x_traj, y_traj, z_traj = sol.y[0], sol.y[1], sol.y[2]

# Plot the x-y projection.
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x_traj, y_traj, label='x-y Trajectory', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('x-y Projection of Trajectory')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
# Draw the plates as before.
ax = plt.gca()
plate_width = 0.1  # thickness in x
plate_height = 2   # extent in y
plate1_center = (-1, 0)
plate2_center = (1, 0)
plate1_ll = (plate1_center[0] - plate_width/2, plate1_center[1] - plate_height/2)
plate2_ll = (plate2_center[0] - plate_width/2, plate2_center[1] - plate_height/2)
rect1 = plt.Rectangle(plate1_ll, plate_width, plate_height, color='gray', alpha=0.5, label='Plate 1')
rect2 = plt.Rectangle(plate2_ll, plate_width, plate_height, color='gray', alpha=0.5, label='Plate 2')
ax.add_patch(rect1)
ax.add_patch(rect2)
plt.legend()

# Plot the z coordinate over time.
plt.subplot(1, 2, 2)
plt.plot(sol.t, z_traj, label='z Position', color='red')
plt.xlabel('Time')
plt.ylabel('z')
plt.title('z Coordinate vs. Time')
plt.legend()

plt.tight_layout()
plt.show()
