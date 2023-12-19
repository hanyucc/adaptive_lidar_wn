import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from mesh import Cube, Cone, Mesh
from camera import Camera

from utils import visualize_points, visualize_lines, estimate_areas, query_ffd, query_occupancy, visualize_ffd, cluster_ray_directions, reconstruct_mesh



camera_views = [
    ('x', 0),
    ('x', 90),
    ('x', 180),
    ('x', -90),
    ('y', 90),
    ('y', -90),
]


def uniform_sampling(mesh, num_samples=16*16*6):
    num_samples_per_view = num_samples // len(camera_views)
    num_samples_per_dim = int(num_samples_per_view ** (1/2))

    points = []
    normals = []

    for axis, angle in camera_views:
        camera = Camera(axis, angle)
        o, d = camera.sample_rays((num_samples_per_dim, num_samples_per_dim))
        hit, p, n = mesh.intersect_rays(o, d)
        points.append(p[hit == 1])
        normals.append(n[hit == 1])

    points = np.concatenate(points, axis=0)
    normals = np.concatenate(normals, axis=0)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # visualize_points(points, ax)
    # ax.view_init(elev=85, azim=90, roll=180)
    # ax.set_xlim(-0.7, 0.7)
    # ax.set_ylim(-0.7, 0.7)
    # ax.set_zlim(-0.7, 0.7)
    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()

    return points, normals


def adaptive_sampling(mesh, num_samples=16*16*6, num_virtual_samples=128*128,
                      tmin=0, tmax=2, tsamples=256):
    initial_samples = num_samples // 4
    points, normals = uniform_sampling(mesh, initial_samples)

    num_samples_per_round = num_samples // 8
    num_rounds = (num_samples - initial_samples) // num_samples_per_round

    num_virtual_samples_per_dim = int(num_virtual_samples ** (1/2))

    for i in range(num_rounds):
        areas = estimate_areas(points)

        ffds = []
        origins = []
        directions = []

        # grid = np.stack(np.meshgrid(
        #     np.linspace(-0.6, 0.4, 100),
        #     np.linspace(0, 1, 100),
        #     np.linspace(0.1, 0.1, 1),
        #     indexing='ij'
        # ), axis=-1).reshape(-1, 3)
        # occu = query_occupancy(points, normals, areas, grid, 10)
        # plt.imshow(occu.reshape(100, 100).transpose(1, 0)[::-1])
        # plt.colorbar()
        # plt.show()

        for axis, angle in camera_views:
            camera = Camera(axis, angle)
            o, d = camera.sample_rays((num_virtual_samples_per_dim, num_virtual_samples_per_dim))

            alpha, tr, ffd = query_ffd(points, areas, normals, o, d, tmin, tmax, tsamples)

            # if (axis, angle) == ('x', 0):
            #     visualize_ffd(alpha, tr, ffd, num_virtual_samples_per_dim, num_virtual_samples_per_dim)

            ffds.append(ffd)
            origins.append(o)
            directions.append(d)

        ffds = np.concatenate(ffds, axis=0)
        origins = np.concatenate(origins, axis=0)
        directions = np.concatenate(directions, axis=0)

        # ffds_normalized = ffds / (ffds.sum(axis=-1)[:, None] + 1e-8)
        ffds_sum = ffds.sum(axis=-1)
        ffds_normalized = np.insert(ffds, -1, 1 - ffds_sum, axis=-1)

        entropy = -(ffds_normalized * np.log(ffds_normalized + 1e-8)).sum(axis=-1)
        candidate_indices = entropy >= np.percentile(entropy, 90)

        o_new = origins[candidate_indices]
        d_new = directions[candidate_indices]

        o_new_clustered = []
        d_new_clustered = []

        o_new_unique, unique_inverse, unique_counts = np.unique(o_new, axis=0, return_inverse=True, return_counts=True)
        for j in range(len(o_new_unique)):
            num_samples_curr_cluster = int(unique_counts[j] / len(o_new) * num_samples_per_round)
            if num_samples_curr_cluster == 0:
                continue

            d_view = d_new[unique_inverse == j]
            d_view_clustered = cluster_ray_directions(d_view, num_samples_curr_cluster)
            o_view_clustered = np.repeat(o_new_unique[j][None, :], d_view_clustered.shape[0], axis=0)

            o_new_clustered.append(o_view_clustered)
            d_new_clustered.append(d_view_clustered)

        o_new_clustered = np.concatenate(o_new_clustered, axis=0)
        d_new_clustered = np.concatenate(d_new_clustered, axis=0)

        hit, p, n = mesh.intersect_rays(o_new_clustered, d_new_clustered)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        visualize_points(points, ax)
        visualize_points(p[hit == 1], ax)
        ax.view_init(elev=85, azim=90, roll=180)
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_zlim(-0.7, 0.7)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

        points = np.concatenate((points, p[hit == 1]), axis=0)
        normals = np.concatenate((normals, n[hit == 1]), axis=0)

        points, unique_indices = np.unique(points, axis=0, return_index=True)
        normals = normals[unique_indices]

    return points, normals


def test_cube():
    cube = Cube(side_length=0.8)

    points, normals = uniform_sampling(cube, 16*16*6)
    visualize_points(points)
    plt.show()

    mesh = reconstruct_mesh(points, normals)
    o3d.io.write_triangle_mesh('results/uniform_cube.obj', mesh)

    print('uniform Chamfer distance:', cube.compute_chamfer_distance('results/uniform_cube.obj', 1 << 20))

    points, normals = adaptive_sampling(cube, 16*16*6)
    visualize_points(points)
    plt.show()

    mesh = reconstruct_mesh(points, normals)
    o3d.io.write_triangle_mesh('results/adaptive_cube.obj', mesh)

    print('adaptive Chamfer distance:', cube.compute_chamfer_distance('results/adaptive_cube.obj', 1 << 20))


def test_cone():
    cone = Cone(radius=0.5, height=1)

    points, normals = uniform_sampling(cone, 16*16*6)
    visualize_points(points)
    plt.show()

    mesh = reconstruct_mesh(points, normals)
    o3d.io.write_triangle_mesh('results/uniform_cone.obj', mesh)

    print('uniform Chamfer distance:', cone.compute_chamfer_distance('results/uniform_cone.obj', 1 << 20))

    points, normals = adaptive_sampling(cone, 16*16*6)
    visualize_points(points)
    plt.show()

    mesh = reconstruct_mesh(points, normals)
    o3d.io.write_triangle_mesh('results/adaptive_cone.obj', mesh)

    print('adaptive Chamfer distance:', cone.compute_chamfer_distance('results/adaptive_cone.obj', 1 << 20))


def test_bunny():
    bunny = Mesh('meshes/bunny.obj')

    points, normals = uniform_sampling(bunny, 32*32*6)
    visualize_points(points)
    plt.show()

    mesh = reconstruct_mesh(points, normals)
    o3d.io.write_triangle_mesh('results/uniform_bunny.obj', mesh)

    print('uniform Chamfer distance:', bunny.compute_chamfer_distance('results/uniform_bunny.obj', 1 << 20))

    points, normals = adaptive_sampling(bunny, 32*32*6)
    visualize_points(points)
    plt.show()

    mesh = reconstruct_mesh(points, normals)
    o3d.io.write_triangle_mesh('results/adaptive_bunny.obj', mesh)

    print('adaptive Chamfer distance:', bunny.compute_chamfer_distance('results/adaptive_bunny.obj', 1 << 20))


def test_buddha():
    buddha = Mesh('meshes/buddha.obj')

    points, normals = uniform_sampling(buddha, 32*32*6)
    visualize_points(points)
    plt.show()

    mesh = reconstruct_mesh(points, normals)
    o3d.io.write_triangle_mesh('results/uniform_buddha.obj', mesh)

    print('uniform Chamfer distance:', buddha.compute_chamfer_distance('results/uniform_buddha.obj', 1 << 20))

    points, normals = adaptive_sampling(buddha, 32*32*6)
    visualize_points(points)
    plt.show()

    mesh = reconstruct_mesh(points, normals)
    o3d.io.write_triangle_mesh('results/adaptive_buddha.obj', mesh)

    print('adaptive Chamfer distance:', buddha.compute_chamfer_distance('results/adaptive_buddha.obj', 1 << 20))



def test_buddha_ablations(l=32):
    buddha = Mesh('meshes/buddha.obj')

    points, normals = uniform_sampling(buddha, l*l*6)
    visualize_points(points)
    plt.show()

    mesh = reconstruct_mesh(points, normals)
    o3d.io.write_triangle_mesh(f'results/ablations/uniform_buddha_{l}.obj', mesh)

    print('uniform Chamfer distance:', buddha.compute_chamfer_distance(f'results/ablations/uniform_buddha_{l}.obj', 1 << 20))

    points, normals = adaptive_sampling(buddha, l*l*6)
    visualize_points(points)
    plt.show()

    mesh = reconstruct_mesh(points, normals)
    o3d.io.write_triangle_mesh(f'results/ablations/adaptive_buddha_{l}.obj', mesh)

    print('adaptive Chamfer distance:', buddha.compute_chamfer_distance(f'results/ablations/adaptive_buddha_{l}.obj', 1 << 20))


# test_cube()
# test_cone()
test_bunny()
# test_buddha()
    
# for l in [10, 16, 32, 48, 64, 96]:
#     test_buddha_ablations(l)
