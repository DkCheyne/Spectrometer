import cProfile
import tracer  # now you can access all objects from tracer.py
import numpy as np

def main():
    # Define the scence objects
    upperBlock = tracer.MagneticBlock(
    position=[0, 1, 0],           # Centered at (0, 1, 0) in cm
    dimensions=[1, 0.1, 0.1],     # 2 cm long, 0.5 cm tall, assuming 0.5 cm depth
    polarization=[0, 0, 10],      # Y-axis polarization with 0.5T strength
    name = "upperBlock"
    )

    lowerBlock = tracer.MagneticBlock(
    position=[0, -1, 0],          # Centered at (0, -1, 0) in cm
    dimensions=[1, 0.1, 0.11],     # 2 cm long, 0.5 cm tall, assuming 0.5 cm depth
    polarization=[0, 0, 10],    # Y-axis polarization with 0.5T strength
    name = "lowerBlock"
    )


    # Define the particle spectrum
    spectrum = tracer.Spectrum(energy_min=1e6, energy_max=30e6, num_particles=10)


    scene = tracer.Scene()
    source = tracer.Source(np.array([-4.0, 0.0, 0.0]), spectrum)
    scene.add_object(source)
    scene.add_object(upperBlock)
    scene.add_object(lowerBlock)
    #scene.visualize_source()
    #scene.visualize_objects()
    #scene.visualize_magnetic_field()
    scene.visualize_particles()
    scene.visualize_3d()

if __name__ == '__main__':
    cProfile.run('main()', sort='cumtime')
