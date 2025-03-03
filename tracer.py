from abc import ABC, abstractmethod
import numpy as np
import magpylib as magpy
from magpylib.magnet import Cuboid
import matplotlib.pyplot as plt

class Object(ABC):
    """Base class for all scene objects."""
    def __init__(self, position):
        self.position = np.array(position)
    
    @abstractmethod
    def interact(self, particle):
        """Defines how an object interacts with a particle."""
        pass

class Spectrum:
    """Defines a particle spectrum with energy distribution."""
    def __init__(self, energy_min=1e6, energy_max=30e6, num_particles=1000, distribution='guassian', mean_angle = 0, 
                 std_dev_angle = np.pi/18, angle_top_limit = np.pi/2, angle_bottom_limit = -np.pi/2): 
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.num_particles = num_particles
        self.distribution = distribution
        self.mean_angle = mean_angle
        self.std_dev_angle = std_dev_angle
        self.angle_top_limit = angle_top_limit
        self.angle_bottom_limit = angle_bottom_limit

    
    def sample_particle(self, position):
        """Generates a particle with a random energy and initial velocity."""
        energy = np.random.uniform(self.energy_min, self.energy_max)  # Energy in eV
        
        # Convert energy from eV to Joules
        E_J = energy * 1.60218e-19  
        m = 1.67e-27
        c = 3e8
        # Relativistic speed: v = c * sqrt(1 - (m*c^2/(m*c^2 + E_J))^2)
        speed = c * np.sqrt(1 - (m * c**2 / (m * c**2 + E_J))**2)
        
        std_dev = np.pi / 6  # Standard deviation of angle distribution
        mean = 0.0  # Mean of angle distribution
        angle = np.random.normal(mean, std_dev)
        lower_bound = -np.pi / 2  # Minimum angle
        upper_bound = np.pi / 2   # Maximum angle
        angle = np.clip(angle, lower_bound, upper_bound)
        self.angle = angle

        # Include a zero for the z-component to match the 3D position
        velocity = speed * np.array([np.cos(angle), np.sin(angle), 0])
        return Particle(np.random.randint(1e6), energy, velocity, angle, position)
    
    def generate_particles(self, position):
        """Generates a list of particles based on the spectrum."""
        return [self.sample_particle(position) for _ in range(self.num_particles)]

class Source(Object):
    """Generates particle spectra."""
    def __init__(self, position, spectrum):
        super().__init__(position)
        self.spectrum = spectrum
    
    def generate_particles(self):
        """Generates particles based on the defined spectrum."""
        # Pass a copy of the source position so that particles do not share the same reference.
        return self.spectrum.generate_particles(np.copy(self.position))
    
    def interact(self, particle):
        pass  # Sources don’t interact with existing particles

class Aperture(Object):
    """Filters particles based on position or trajectory."""
    def __init__(self, position, radius):
        super().__init__(position)
        self.radius = radius
    
    def interact(self, particle):
        distance = np.linalg.norm(particle.position - self.position)
        if distance > self.radius:
            particle.alive = False  # Remove particles that don’t pass through the aperture

class MagneticBlock(Object):
    """Applies a magnetic field to particles using MagPy."""
    def __init__(self, position, dimensions, polarization, name):
        super().__init__(position)
        self.magnet = Cuboid(polarization=polarization, dimension=dimensions, position=position)
        self.name = name
    
    def interact(self, particle):
        """Applies the Lorentz force to the particle using MagPy."""
        q, v = particle.charge, particle.current_velocity
        B = self.magnet.getB(particle.position)
        lorentz_force = q * np.cross(v, B)
        particle.apply_force(lorentz_force)

class Detector(Object):
    """Records particle data."""
    def __init__(self, position):
        super().__init__(position)
        self.detected_particles = []
    
    def interact(self, particle):
        self.detected_particles.append(particle)

    def get_data(self):
        """Returns recorded particle information."""
        return self.detected_particles

class Particle:
    """Represents a particle with energy conservation."""
    def __init__(self, particle_id, start_energy, start_velocity, angle, position=np.array([0.0, 0.0])):
        self.id = particle_id
        self.start_energy = start_energy
        self.start_velocity = np.array(start_velocity)
        self.current_velocity = np.array(start_velocity)
        # Initialize position using a copy of the provided 'position'
        self.position = np.copy(position)
        self.alive = True
        self.trajectory = [self.position.copy()]
        self.angle = angle  # Angle of particle trajectory
        self.charge = 1.6e-19  # Charge of an electron in Coulombs
    
    def apply_force(self, force, dt=0.5e-9):
        """Applies force while conserving energy."""
        acceleration = force  # Assuming unit mass for simplicity
        self.current_velocity += acceleration * dt
        
        # Conserve energy
        current_speed = np.linalg.norm(self.current_velocity)
        initial_speed = np.linalg.norm(self.start_velocity)
        if current_speed > 0:
            self.current_velocity *= (initial_speed / current_speed)
    
    def update_position(self, dt=0.5e-9):
        """Updates position based on velocity."""
        self.position += self.current_velocity * dt
        self.trajectory.append(self.position.copy())

    def get_trajectory(self):
        """Returns the recorded trajectory of the particle."""
        return np.array(self.trajectory)

def visualize_scene(particles, objects):
    """Plots the scene with particle trajectories, objects, and magnetic field streamlines."""
    plt.figure(figsize=(8, 6))
    
    # Plot particle trajectories
    for particle in particles:
        trajectory = particle.get_trajectory()
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Particle {particle.id}')
    
    # Plot objects
    for obj in objects:
        if isinstance(obj, Source):
            plt.scatter(obj.position[0], obj.position[1], marker='*', color='red', s=100, label='Source')
        elif isinstance(obj, Aperture):
            plt.scatter(obj.position[0], obj.position[1], marker='o', color='green', s=100, label='Aperture')
        elif isinstance(obj, Detector):
            plt.scatter(obj.position[0], obj.position[1], marker='^', color='purple', s=100, label='Detector')
    

    
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Particle Trajectories, Objects, and Magnetic Field Streamlines")
    plt.legend()
    plt.grid()
    plt.show()

def visualize_source(source, num_particles=10):
    """Plots the initial particle trajectories from a source."""
    """There are no other objects in the scene."""
    particles = source.generate_particles()
    plt.figure(figsize=(8, 6))
    
    for particle in particles[:num_particles]:
        trajectory = [particle.position]
        for _ in range(10):
            particle.update_position()
        plt.plot([xy[0] for xy in particle.trajectory], [xy[1] for xy in particle.trajectory], label=f'Particle {particle.id}')
    
    plt.scatter(source.position[0], source.position[1], marker='*', color='red', s=100, label='Source')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Initial Particle Trajectories from Source")
    plt.legend()
    plt.grid()
    plt.show()

    plt.hist([p.angle for p in particles], bins=50, density=True)
    plt.xlabel("Angle")     
    plt.ylabel("Density")
    plt.title("Initial Particle Angle Distribution")
    plt.show()

    plt.hist([p.start_energy for p in particles], bins=50, density=True)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Density")
    plt.title("Initial Particle Energy Distribution")
    plt.show()

class Scene:
    """Manages the entire simulation scene, including objects, particles, interactions, and timesteps."""
    def __init__(self):
        self.objects = []
        self.particles = []
        self.dt = 0.5e-9  # Timestep in seconds
        self.x_min, self.x_max = -5, 5
        self.y_min, self.y_max = -5, 5
        self.resolution = 50
    
    def add_object(self, obj):
        self.objects.append(obj)
    
        if isinstance(obj, Source):
            particles = obj.generate_particles()
            self.particles.extend(particles)

    def run_simulation(self, num_steps=1000, dt=0.5e-9):
        for _ in range(num_steps):
            for obj in self.objects:
                for particle in self.particles:
                    obj.interact(particle)
                    if not particle.alive:
                        #self.particles.remove(particle)
                        pass
                    else:
                        particle.update_position(dt)
                        if particle.position[0] < self.x_min or particle.position[0] > self.x_max or particle.position[1] < self.y_min or particle.position[1] > self.y_max:
                            particle.alive = False

    def visualize_objects(self): 
        """Plots the scene with particle trajectories, objects, and overlays magnetic field streamlines."""
        plt.figure(figsize=(8, 6))
        
        # First plot the streamlines for MagneticBlock contributions
        resolution = 50
        x = np.linspace(self.x_min, self.x_max, resolution)
        y = np.linspace(self.y_min, self.y_max, resolution)
        X, Y = np.meshgrid(x, y)
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)
        for obj in self.objects:
            if isinstance(obj, MagneticBlock):
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        B = obj.magnet.getB([X[i, j], Y[i, j], 0])
                        U[i, j] += B[0]
                        V[i, j] += B[1]
        plt.streamplot(X, Y, U, V, color='blue', density=1.0, linewidth=1, arrowsize=1.5)
        
        # Plot objects after the streamlines so they overlay on top
        for obj in self.objects:
            if isinstance(obj, Source):
                plt.scatter(obj.position[0], obj.position[1], marker='*', color='red', s=100, label='Source')
            elif isinstance(obj, Aperture):
                plt.scatter(obj.position[0], obj.position[1], marker='o', color='green', s=100, label='Aperture')
            elif isinstance(obj, MagneticBlock):
                x_obj, y_obj = obj.magnet.position[:2]
                w, h = obj.magnet.dimension[:2]
                rect = plt.Rectangle((x_obj - w/2, y_obj - h/2), w, h, color='blue', alpha=0.5, label=obj.name)
                plt.gca().add_patch(rect)
            elif isinstance(obj, Detector):
                plt.scatter(obj.position[0], obj.position[1], marker='^', color='purple', s=100, label='Detector')
        
        # Plot particle trajectories (if needed; this block can be uncommented/modified)
        #for particle in self.particles:
        #    trajectory = particle.get_trajectory()
        #    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Particle {particle.id}')
        
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Scene with Magnetic Field Streamlines Overlay")
        plt.legend()
        plt.grid()
        plt.show()

    def visualize_source(self, num_particles=10):
        visualize_source(source, num_particles)


    ###TODO: Implement the following methods
    def visualize_magnetic_field_magpy(self):

        B = block.magnet.getB()
        magpy.displaySystem(block.magnet)
        magpy.displayField(block.magnet, subplot=True)
        plt.show()

    def visualize_magnetic_field(self, resolution=50, z=0):
        """Visualizes the combined magnetic field from all MagneticBlock objects in the scene."""
        # Create grid over scene region
        x = np.linspace(self.x_min, self.x_max, resolution)
        y = np.linspace(self.y_min, self.y_max, resolution)
        X, Y = np.meshgrid(x, y)
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)
        
        # Sum contributions from all MagneticBlock objects
        for obj in self.objects:
            if isinstance(obj, MagneticBlock):
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        # Get the local magnetic field vector at (x,y,z)
                        B = obj.magnet.getB([X[i, j], Y[i, j], z])
                        U[i, j] += B[0]
                        V[i, j] += B[1]
        
        plt.figure(figsize=(8, 6))
        plt.streamplot(X, Y, U, V, color='blue', density=1.0, linewidth=1, arrowsize=1.5, alpha = 0.5)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Magnetic Field Streamlines from Magnetic Blocks")
        plt.grid()
        plt.show()
    

    def visualize_particles(self):

        self.particles = source.generate_particles()

        self.run_simulation(num_steps=1000, dt=0.5e-9)

        """Plots the scene with particle trajectories, objects, and magnetic field streamlines."""
        plt.figure(figsize=(8, 6))
        
        # Plot particle trajectories
        for particle in self.particles:
            trajectory = particle.get_trajectory()
            plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Particle {particle.id}')
            #plt.show()
        
        # Plot objects
        for obj in self.objects:
            if isinstance(obj, Source):
                plt.scatter(obj.position[0], obj.position[1], marker='*', color='red', s=100, label='Source')
            elif isinstance(obj, Aperture):
                plt.scatter(obj.position[0], obj.position[1], marker='o', color='green', s=100, label='Aperture')
            elif isinstance(obj, MagneticBlock):
                x_obj, y_obj = obj.magnet.position[:2]
                w, h = obj.magnet.dimension[:2]
                rect = plt.Rectangle((x_obj - w/2, y_obj - h/2), w, h, color='blue', alpha=0.5, label=obj.name)
                plt.gca().add_patch(rect)
            elif isinstance(obj, Detector):
                plt.scatter(obj.position[0], obj.position[1], marker='^', color='purple', s=100, label='Detector')


        # Create grid over scene region
        x = np.linspace(self.x_min, self.x_max, self.resolution)
        y = np.linspace(self.y_min, self.y_max, self.resolution)
        X, Y = np.meshgrid(x, y)
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)
        z = 0
            # Sum contributions from all MagneticBlock objects
        for obj in self.objects:
            if isinstance(obj, MagneticBlock):
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        # Get the local magnetic field vector at (x,y,z)
                        B = obj.magnet.getB([X[i, j], Y[i, j], z])
                        U[i, j] += B[0]
                        V[i, j] += B[1]
        
        plt.streamplot(X, Y, U, V, color='blue', density=1.0, linewidth=1, arrowsize=1.5)
        
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Particle Trajectories, Objects, and Magnetic Field Streamlines")
        plt.legend()
        plt.grid()
        plt.show()
    

# Define the scence objects
upperBlock = MagneticBlock(
    position=[0, 1, 0],           # Centered at (0, 1, 0) in cm
    dimensions=[2, 0.5, 0.5],     # 2 cm long, 0.5 cm tall, assuming 0.5 cm depth
    polarization=[0, 0.5, 0],      # Y-axis polarization with 0.5T strength
    name = "upperBlock"
)

lowerBlock = MagneticBlock(
    position=[0, -1, 0],          # Centered at (0, -1, 0) in cm
    dimensions=[2, 0.5, 0.5],     # 2 cm long, 0.5 cm tall, assuming 0.5 cm depth
    polarization=[0, 0.5, 0],    # Y-axis polarization with 0.5T strength
    name = "lowerBlock"
)


# Define the particle spectrum
spectrum = Spectrum(energy_min=1e6, energy_max=30e6, num_particles=10)


scene = Scene()
source = Source(np.array([-4.0, 0.0, 0.0]), spectrum)
scene.add_object(source)
scene.add_object(upperBlock)
scene.add_object(lowerBlock)
#scene.visualize_source()
#scene.visualize_objects()
#scene.visualize_magnetic_field()
scene.visualize_particles()
