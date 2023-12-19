import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh("vis/results/buddha.obj")
# mesh = o3d.io.read_triangle_mesh("results/uniform_buddha.obj")
# mesh = o3d.io.read_triangle_mesh("results/ablations/adaptive_buddha_96.obj")
mesh.compute_triangle_normals()
mesh.paint_uniform_color([0.5, 0.5, 0.5])

camera1 = o3d.geometry.TriangleMesh.create_sphere(0.01, 20)
camera1.translate(np.array([0, 0, 1.5]))
camera1.compute_triangle_normals()
camera1.paint_uniform_color([0.1, 0.1, 0.1])

box1 = o3d.geometry.TriangleMesh.create_box(0.5, 0.5, 0.01)
box1.translate(np.array([-0.25, -0.25, 1.1]))
box1.compute_triangle_normals()
box1.paint_uniform_color([0, 0.5, 0.8])

camera2 = o3d.geometry.TriangleMesh.create_sphere(0.01, 20)
camera2.translate(np.array([0, 0, -1.5]))
camera2.compute_triangle_normals()
camera2.paint_uniform_color([0.1, 0.1, 0.1])

box2 = o3d.geometry.TriangleMesh.create_box(0.5, 0.5, 0.01)
box2.translate(np.array([-0.25, -0.25, -1.1]))
box2.compute_triangle_normals()
box2.paint_uniform_color([0, 0.5, 0.8])

camera3 = o3d.geometry.TriangleMesh.create_sphere(0.01, 20)
camera3.translate(np.array([0, 1.5, 0]))
camera3.compute_triangle_normals()
camera3.paint_uniform_color([0.1, 0.1, 0.1])

box3 = o3d.geometry.TriangleMesh.create_box(0.5, 0.01, 0.5)
box3.translate(np.array([-0.25, 1.1, -0.25]))
box3.compute_triangle_normals()
box3.paint_uniform_color([0, 0.5, 0.8])

camera4 = o3d.geometry.TriangleMesh.create_sphere(0.01, 20)
camera4.translate(np.array([0, -1.5, 0]))
camera4.compute_triangle_normals()
camera4.paint_uniform_color([0.1, 0.1, 0.1])

box4 = o3d.geometry.TriangleMesh.create_box(0.5, 0.01, 0.5)
box4.translate(np.array([-0.25, -1.1, -0.25]))
box4.compute_triangle_normals()
box4.paint_uniform_color([0, 0.5, 0.8])

camera5 = o3d.geometry.TriangleMesh.create_sphere(0.01, 20)
camera5.translate(np.array([1.5, 0, 0]))
camera5.compute_triangle_normals()
camera5.paint_uniform_color([0.1, 0.1, 0.1])

box5 = o3d.geometry.TriangleMesh.create_box(0.01, 0.5, 0.5)
box5.translate(np.array([1.1, -0.25, -0.25]))
box5.compute_triangle_normals()
box5.paint_uniform_color([0, 0.5, 0.8])

camera6 = o3d.geometry.TriangleMesh.create_sphere(0.01, 20)
camera6.translate(np.array([-1.5, 0, 0]))
camera6.compute_triangle_normals()
camera6.paint_uniform_color([0.1, 0.1, 0.1])

box6 = o3d.geometry.TriangleMesh.create_box(0.01, 0.5, 0.5)
box6.translate(np.array([-1.1, -0.25, -0.25]))
box6.compute_triangle_normals()
box6.paint_uniform_color([0, 0.5, 0.8])

mat_sphere = o3d.visualization.rendering.MaterialRecord()
mat_sphere.shader = 'defaultLit'
mat_sphere.base_color = [0.5, 0.5, 0.5, 1.0]

mat_box = o3d.visualization.rendering.MaterialRecord()
mat_box.shader = 'defaultLitTransparency'
mat_box.base_color = [0.4, 0.4, 0.8, 0.6]
mat_box.base_roughness = 1.0
mat_box.base_reflectance = 0.0
mat_box.base_clearcoat = 0.0
mat_box.thickness = 1.0
mat_box.transmission = 1.0
mat_box.absorption_distance = 10
mat_box.absorption_color = [0.5, 0.5, 0.5]

points = [
    [1.5, 0, 0],
    [-1.5, 0, 0],
    [0, 1.5, 0],
    [0, -1.5, 0],
    [0, 0, 1.5],
    [0, 0, -1.5],
    [1.1, 0.25, 0.25],
    [1.1, 0.25, -0.25],
    [1.1, -0.25, 0.25],
    [1.1, -0.25, -0.25],
    [-1.1, 0.25, 0.25],
    [-1.1, 0.25, -0.25],
    [-1.1, -0.25, 0.25],
    [-1.1, -0.25, -0.25],
    [0.25, 1.1, 0.25],
    [0.25, 1.1, -0.25],
    [-0.25, 1.1, 0.25],
    [-0.25, 1.1, -0.25],
    [0.25, -1.1, 0.25],
    [0.25, -1.1, -0.25],
    [-0.25,-1.1, 0.25],
    [-0.25, -1.1, -0.25],
    [0.25, 0.25, 1.1],
    [0.25, -0.25, 1.1],
    [-0.25, 0.25, 1.1],
    [-0.25, -0.25, 1.1],
    [0.25, 0.25, -1.1],
    [0.25, -0.25, -1.1],
    [-0.25, 0.25, -1.1],
    [-0.25, -0.25, -1.1],
]
lines = [
    [0, 6], [0, 7], [0, 8], [0, 9],
    [1, 10], [1, 11], [1, 12], [1, 13],
    [2, 14], [2, 15], [2, 16], [2, 17],
    [3, 18], [3, 19], [3, 20], [3, 21],
    [4, 22], [4, 23], [4, 24], [4, 25],
    [5, 26], [5, 27], [5, 28], [5, 29],
]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)

geoms = [{'name': 'box', 'geometry': mesh, 'material': mat_sphere},
         {'name': 'image1', 'geometry': box1, 'material': mat_box},
         {'name': 'camera1', 'geometry': camera1, 'material': mat_sphere},
         {'name': 'image2', 'geometry': box2, 'material': mat_box},
         {'name': 'camera2', 'geometry': camera2, 'material': mat_sphere},
         {'name': 'image3', 'geometry': box3, 'material': mat_box},
         {'name': 'camera3', 'geometry': camera3, 'material': mat_sphere},
         {'name': 'image4', 'geometry': box4, 'material': mat_box},
         {'name': 'camera4', 'geometry': camera4, 'material': mat_sphere},
         {'name': 'image5', 'geometry': box5, 'material': mat_box},
         {'name': 'camera5', 'geometry': camera5, 'material': mat_sphere},
         {'name': 'image6', 'geometry': box6, 'material': mat_box},
         {'name': 'camera6', 'geometry': camera6, 'material': mat_sphere},
         {'name': 'lines', 'geometry': line_set, 'material': mat_sphere}]

o3d.visualization.draw(geoms, eye=[0, 0, -2], lookat=[0, 0, 0], up=[0, 1, 0], show_skybox=False, bg_color=[1, 1, 1, 1], show_ui=True)
