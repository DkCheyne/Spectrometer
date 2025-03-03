import magpylib as magpy
import numpy as np
import matplotlib.pyplot as plt


# Define two parallel plate magnets as thin cuboids.
# Their large faces lie in the y–z plane.
plate1 = magpy.magnet.Cuboid(
    polarization=(0, 0.5, 1),    # magnetized in +x direction
    dimension=(2, 0.1, 1),      # thin plate: 0.1 unit thick, 2 units in y, 1 unit in z
    position=(0, -1, 0)         # positioned at x = -1
)

plate2 = magpy.magnet.Cuboid(
    polarization=(0, 0.5, 0),   # magnetized in -x direction
    dimension=(2, 0.1, 1),
    position=(0, 1, 0)          # positioned at x = 1
)

# Create an observation grid in the x–y plane at z = 0.
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
points = np.column_stack((X.flatten(), Y.flatten(), np.zeros(X.size)))

# Calculate the magnetic field from each plate and sum them.
B1 = plate1.getB(points)
B2 = plate2.getB(points)
B_total = B1 + B2

# Reshape the field components to match the grid.
B_total_x = B_total[:, 0].reshape(X.shape)
B_total_y = B_total[:, 1].reshape(X.shape)

# Set up the plot.
plt.figure(figsize=(8, 6))

# Plot the magnetic field lines.
plt.streamplot(X, Y, B_total_x, B_total_y, density=1.5, linewidth=1, arrowsize=1, arrowstyle='->')

# Draw the two plates as rectangles.
ax = plt.gca()

# Plate dimensions in the x-y plane:
# thickness in x: 0.1, extent in y: 2 (centered at y=0)
# For a plate centered at (x0, y0), the lower left corner is (x0 - 0.05, y0 - 1).
plate_width = 2
plate_height = 0.1
plate1_center = (0, -1)
plate2_center = (0, 1)

# Calculate lower-left corners.
plate1_ll = (plate1_center[0] - plate_width/2, plate1_center[1] - plate_height/2)
plate2_ll = (plate2_center[0] - plate_width/2, plate2_center[1] - plate_height/2)

# Create and add rectangle patches.
rect1 = plt.Rectangle(plate1_ll, plate_width, plate_height, color='gray', alpha=0.5, label='Plate 1')
rect2 = plt.Rectangle(plate2_ll, plate_width, plate_height, color='gray', alpha=0.5, label='Plate 2')
ax.add_patch(rect1)
ax.add_patch(rect2)

# Optionally, add a legend for the plates.
plt.legend(handles=[rect1, rect2], loc='upper right')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Basic Magnetic Configuration')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
