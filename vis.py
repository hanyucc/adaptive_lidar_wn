import open3d as o3d

# mesh = o3d.io.read_triangle_mesh("vis/results/buddha.obj")
# mesh = o3d.io.read_triangle_mesh("results/uniform_buddha.obj")

mesh = o3d.io.read_triangle_mesh("results/ablations/adaptive_buddha_96.obj")
mesh.compute_triangle_normals()
mesh.paint_uniform_color([0.5, 0.5, 0.5])

mat_sphere = o3d.visualization.rendering.MaterialRecord()
mat_sphere.shader = 'defaultLit'
mat_sphere.base_color = [0.5, 0.5, 0.5, 1.0]

geoms = [{'name': 'box', 'geometry': mesh, 'material': mat_sphere}]

o3d.visualization.draw(geoms, eye=[0, 0, -2], lookat=[0, 0, 0], up=[0, 1, 0], show_skybox=False, bg_color=[1, 1, 1, 1], show_ui=True)
