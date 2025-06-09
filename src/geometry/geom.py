import numpy as np
import pyvista as pv # Import PyVista

class Geometry3D:
    def __init__(self, center=(0,0,0), quality='medium', rotation=(0,0,0)):
        self.center = np.array(center, dtype=float)
        self.quality = quality  # 'low', 'medium', 'high'
        self.rotation = np.array(rotation, dtype=float)  # Euler angles in degrees (rx, ry, rz)
        self.vertices = np.array([]) # Initialize as empty numpy array
        self.faces = []     # List of faces (indices into vertices)
        self.params = {}    # Custom parameters (radius, width, etc)
        self.mesh = None    # Add mesh attribute

    def get_data(self):
        return {
            'center': self.center,
            'quality': self.quality,
            'rotation': self.rotation,
            'vertices': self.vertices,
            'faces': self.faces,
            'params': self.params,
            'mesh': self.mesh # Include mesh in data if needed
        }

class Sphere(Geometry3D):
    def __init__(self, center=(0,0,0), radius=1.0, quality='medium', rotation=(0,0,0)):
        super().__init__(center, quality, rotation)
        self.params['radius'] = radius
        
        # Create PyVista mesh and populate vertices
        quality_map = {'low': 8, 'medium': 16, 'high': 32, 'ultra': 64} # Added ultra
        resolution = quality_map.get(self.quality, 16)
        
        pv_mesh = pv.Sphere(
            center=tuple(self.center), # Ensure center is a tuple
            radius=self.params['radius'],
            theta_resolution=resolution,
            phi_resolution=resolution
        )
        self.mesh = pv_mesh
        self.vertices = np.array(self.mesh.points, dtype=np.float32)

class Cube(Geometry3D):
    def __init__(self, center=(0,0,0), width=1.0, quality='low', rotation=(0,0,0)):
        super().__init__(center, quality, rotation)
        self.params['width'] = width

        # Map quality to subdivisions (number of divisions per axis)
        quality_map = {'low': 1, 'medium': 2, 'high': 3, 'ultra': 8}
        subdivisions = quality_map.get(self.quality, 4)

        # Use pyvista's Box and subdivide filter to tessellate the cube
        pv_mesh = pv.Box(
            bounds=(
                self.center[0] - width/2, self.center[0] + width/2,
                self.center[1] - width/2, self.center[1] + width/2,
                self.center[2] - width/2, self.center[2] + width/2
            )
        )
        if subdivisions > 1:
            pv_mesh.triangulate(inplace=True) # Ensure mesh is triangulated before subdivision
            pv_mesh = pv_mesh.subdivide(subdivisions, subfilter='linear')

        self.mesh = pv_mesh
        self.vertices = np.array(self.mesh.points, dtype=np.float32)

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
