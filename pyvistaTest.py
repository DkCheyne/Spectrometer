import astropy.units as u
import numpy as np

from psipy.data import sample_data
from psipy.model import MASOutput
from psipy.tracing import FortranTracer
from psipy.visualization.pyvista import PyvistaPlotter
mas_path = sample_data.mas_sample_data()
model = MASOutput(mas_path)
tracer = FortranTracer()

nseeds = 20
r = np.ones(nseeds**2) * 40 * u.R_sun
lat = np.linspace(-45, 45, nseeds**2, endpoint=False) * u.deg
lon = np.random.rand(nseeds**2) * 360 * u.deg

# Do the tracing!
flines = tracer.trace(model, r=r, lat=lat, lon=lon)
plotter = PyvistaPlotter(model)
br = model["br"]
for fline in flines:
    # Set color with polarity near the inner boundary
    color = (
        br.sample_at_coords(
            np.mod(fline.lon[1], 2 * np.pi * u.rad),
            fline.lat[1],
            fline.r[1],
        )
        > 0
    )
    color = {0: "red", 1: "blue"}[int(color)]
    plotter.add_fline(fline, color=color)

# Add a sphere at the inner boundary of the model
plotter.add_sphere(np.min(br.r_coords))
plotter.show()