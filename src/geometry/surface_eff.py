import numpy as np
import matplotlib.pyplot as plt
from geometry.render3d import Render3D
from geometry.geom import Sphere, Cube

class Surface_Noise:


    def perlin_noise(self, x, y, z, scale=1.0, octaves=1, persistence=0.5, lacunarity=2.0):
        """Generate Perlin noise for a 3D surface."""
        