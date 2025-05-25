import sys
import os
from geom import Cube, Sphere, Triangle, ExoticFormulaShape
from render3d import Render3D, Render2D
import cv2

# 3D Render Test
r3d = Render3D(img_size=512, background='white')
# Helper: minimum separation for no overlap
R = 0.7
min_sep = 2 * R + 0.1  # add a small gap
positions = [
    (0, 0, 4),
    (3, 0, 2.5),
    (-3, 0, 2.5),
    (0, 3, 1.5),
    (0, -3, 1.5),
    (3, 3, -2),
    (-3, -3, 0),
    (3, -3, 0.5),
    (-3, 3, 4),
    (2, -4, -3),
    (-2, 4, -2.5),
    (4, 2, 3),
    (-4, -2, 3),
    (-4, 4, -3),
    (4, -4, -3)
]
colors = [
    'white', 'red', 'blue', 'orange', 'cyan', 'green', 'purple', 'yellow', 'magenta', 'pink', 'lime',
    'gold', 'violet', 'brown', 'teal'
]
rotations = [
    (0,0,0), (0,0,0), (0,0,0), (45,0,0), (0,45,0), (0,0,45), (0,10,0), (0,0,10), (10,0,0), (0,0,0), (0,0,0),
    (20,10,0), (0,0,0), (0,30,10), (0,0,0)
]
# Alternate between spheres and cubes
for i, pos in enumerate(positions):
    if i % 2 == 0:
        r3d.add_geometry(Sphere(center=pos, radius=R, quality='high', rotation=rotations[i]), color=colors[i], opacity=1)
    else:
        r3d.add_geometry(Cube(center=pos, width=R, quality='medium', rotation=rotations[i]), color=colors[i], opacity=1)

# Add a triangle (not a volumetric object, but for completeness)
r3d.add_geometry(Triangle((2,2,2), (2.5,2,2), (2,2.5,2), rotation=(0,0,0)), color='black', opacity=1)
# Add an exotic formula shape (placeholder, no formula)
r3d.add_geometry(ExoticFormulaShape(center=(0,0,-2), formula=None, quality='medium', rotation=(0,0,0)), color='gray', opacity=1)
# Add a sphere at the center of the perspective, close to the camera
r3d.add_geometry(Sphere(center=(0,0,4), radius=0.7, quality='high', rotation=(0,0,0)), color='white', opacity=1)  # center, very close

# Additional geometries for a richer scene
r3d.add_geometry(Cube(center=(3,3,3), width=0.7, quality='medium', rotation=(20,10,0)), color='gold', opacity=1)          # scattered
r3d.add_geometry(Sphere(center=(-3,-3,3), radius=0.7, quality='high', rotation=(0,0,0)), color='violet', opacity=1)      # scattered
r3d.add_geometry(Cube(center=(-3,3,-3), width=0.7, quality='medium', rotation=(0,30,10)), color='brown', opacity=1)      # scattered
r3d.add_geometry(Sphere(center=(3,-3,-3), radius=0.7, quality='high', rotation=(0,0,0)), color='teal', opacity=1)        # scattered

# For best depth, set a good camera position:
camera_position = [(10, 10, 10), (0, 0, 0), (0, 0, 5)]  # (eye, target, up)
r3d.render(output='test_render3d.png', show_edges=False, camera_position=camera_position)

# 2D Render Test
r2d = Render2D(img_size=512, background='white')
r2d.add_geometry(Sphere(center=(0,0,0), radius=0.6, quality='high', rotation=(0,0,0)), color='red', linewidth=2)
r2d.add_geometry(Cube(center=(1,1,0), width=0.6, quality='medium', rotation=(0,0,0)), color='blue', linewidth=2)
r2d.add_geometry(Triangle((0,0,0), (1,0,0), (0,1,0), rotation=(0,0,0)), color='magenta', linewidth=2)
r2d.render(output='test_render2d.png')

# Show results with OpenCV
img3d = cv2.imread('test_render3d.png')
img2d = cv2.imread('test_render2d.png')
cv2.imshow('3D Render', img3d)
cv2.imshow('2D Render', img2d)
cv2.waitKey(0)
cv2.destroyAllWindows()
