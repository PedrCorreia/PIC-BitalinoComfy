import numpy as np

class Geometry3D:
    def __init__(self, center=(0,0,0), quality='medium', rotation=(0,0,0)):
        self.center = np.array(center, dtype=float)
        self.quality = quality  # 'low', 'medium', 'high'
        self.rotation = np.array(rotation, dtype=float)  # Euler angles in degrees (rx, ry, rz)
        self.vertices = []  # List of 3D points
        self.faces = []     # List of faces (indices into vertices)
        self.params = {}    # Custom parameters (radius, width, etc)

    def get_data(self):
        return {
            'center': self.center,
            'quality': self.quality,
            'rotation': self.rotation,
            'vertices': self.vertices,
            'faces': self.faces,
            'params': self.params
        }

class Sphere(Geometry3D):
    def __init__(self, center=(0,0,0), radius=1.0, quality='medium', rotation=(0,0,0)):
        super().__init__(center, quality, rotation)
        self.params['radius'] = radius
        # Optionally: generate vertices/faces for mesh here

class Cube(Geometry3D):
    def __init__(self, center=(0,0,0), width=1.0, quality='medium', rotation=(0,0,0)):
        super().__init__(center, quality, rotation)
        self.params['width'] = width
        # Optionally: generate vertices/faces for mesh here

class Triangle(Geometry3D):
    def __init__(self, v1, v2, v3, quality='medium', rotation=(0,0,0)):
        super().__init__(center=np.mean([v1, v2, v3], axis=0), quality=quality, rotation=rotation)
        self.vertices = [np.array(v1), np.array(v2), np.array(v3)]
        self.faces = [(0,1,2)]

class ExoticFormulaShape(Geometry3D):
    def __init__(self, center=(0,0,0), formula=None, quality='medium', rotation=(0,0,0)):
        super().__init__(center, quality, rotation)
        self.params['formula'] = formula
        # Optionally: generate vertices/faces from formula
