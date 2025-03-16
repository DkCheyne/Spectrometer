import numpy as np
import magpylib as magpy
from streamtracer import StreamTracer, VectorGrid
import concurrent.futures
import pyvista as pv

# Global grid and tracer settings
nsteps = 1000
step_size = 0.01
tracer = StreamTracer(nsteps, step_size)
ngrid = 20


x_range = np.linspace(-5, 5, ngrid)
y_range = np.linspace(-5, 5, ngrid)
z_range = np.linspace(-5, 5, ngrid)

x_spacing = (x_range[-1] - x_range[0]) / (len(x_range) - 1)
y_spacing = (y_range[-1] - y_range[0]) / (len(y_range) - 1)
z_spacing = (z_range[-1] - z_range[0]) / (len(z_range) - 1)

field = np.empty((ngrid, ngrid, ngrid, 3))

# Initializer for each process to create a global magnet
def init_worker():
    global magnet_worker
    magnet_worker = magpy.magnet.Cuboid(polarization=[0, 0, 10], dimension=[1, 1, 1], position=[0, 0, 0])

def compute_field(indices):
    i, j, k = indices
    x = x_range[i]
    y = y_range[j]
    z = z_range[k]
    B = magnet_worker.getB([x, y, z])
    return (i, j, k, B)

def main():
    indices_list = [(i, j, k) for i in range(ngrid) for j in range(ngrid) for k in range(ngrid)]

    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:
        results = executor.map(compute_field, indices_list)
        for i, j, k, B in results:
            field[i, j, k, :] = B

    vector_spacing = [x_spacing, y_spacing, z_spacing]
    vector_grid = VectorGrid(field, vector_spacing)

    # Define seed points for streamlines (circle in x-y plane)
    num_seeds = 10
    radius = 1.25
    theta = np.linspace(0, 2 * np.pi, num_seeds, endpoint=False)
    seeds = np.stack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.ones(num_seeds)*-2
    ], axis=1)

    tracer.trace(seeds, vector_grid)
    
    

    # Plotting with pyvista
    plotter = pv.Plotter()
    for streamline in tracer.xs:
        points = np.array(streamline)
        n_points = points.shape[0]
        cells = np.hstack(([n_points], np.arange(n_points)))
        polyline = pv.PolyData(points, lines=cells)
        plotter.add_mesh(polyline, color="blue", line_width=2)

    magnet_cube = pv.Cube(center=[0, 0, 0], x_length=1, y_length=1, z_length=1)
    
    
    # Add coordinate axes lines manually
    # X-axis: red, Y-axis: green, Z-axis: blue
    x_line = pv.Line(pointa=[-6, 0, 0], pointb=[6, 0, 0])
    y_line = pv.Line(pointa=[0, -6, 0], pointb=[0, 6, 0])
    z_line = pv.Line(pointa=[0, 0, -6], pointb=[0, 0, 6])
    plotter.add_mesh(x_line, color="red", line_width=3)
    plotter.add_mesh(y_line, color="green", line_width=3)
    plotter.add_mesh(z_line, color="blue", line_width=3)

    plotter.add_mesh(magnet_cube, color="red", opacity=0.5)
    plotter.show()

if __name__ == '__main__':
    main()
