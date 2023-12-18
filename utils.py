import igl
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import open3d as o3d


def estimate_areas(points):
    pca = PCA(2)
    knn = NearestNeighbors(n_neighbors=10).fit(points)

    indices = knn.kneighbors(return_distance=False)

    def worker(i):
        idx = indices[i]
        idx = np.insert(idx, 0, i)

        nbs = points[idx]
        nbs_new = pca.fit_transform(nbs)

        try:
            vor = Voronoi(nbs_new)
        except Exception as e:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(nbs[:, 0], nbs[:, 1], nbs[:, 2])
            plt.show()

        try:
            reg = vor.regions[vor.point_region[0]]
            if -1 in reg:
                return 0.
            hull = ConvexHull(vor.vertices[reg])
        except Exception as e:
            return 0.

        area = hull.volume
        return area

    print(f'Estimating areas for {len(points)} points')
    # areas = Parallel(32, verbose=5)(delayed(worker)(i) for i in range(len(points)))
    areas = [worker(i) for i in range(len(points))]
    areas = np.array(areas).reshape(-1, 1)

    perc_95 = np.percentile(areas, 95)
    areas = np.clip(areas, 0., perc_95)

    return areas


def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


def stable_sigmoid(x):
    result = np.zeros_like(x)
    result[x > 100] = 1
    result[x < -100] = -1
    valid_range = np.logical_and(x >= -100, x <= 100)
    result[valid_range] = sigmoid(x[valid_range])
    return result


def visualize_lines(o, d, tmin, tmax, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    o = o.reshape(-1, 3)
    d = d.reshape(-1, 3)

    start = o + d * tmin
    end = o + d * tmax

    for a, b in zip(start, end):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 'r-', linewidth=0.1)


def visualize_points(p, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=0.5)


def query_occupancy(p, n, a, q, s=10):
    wn = igl.fast_winding_number_for_points(p, n, a, q)
    occu = stable_sigmoid((wn - 1/2) * s)
    return occu


def query_ffd(p, a, n, o, d, tmin=0, tmax=10, tsamples=512):
    o = o.reshape(-1, 3)
    d = d.reshape(-1, 3)

    ts = np.linspace(tmin, tmax, tsamples)
    q = o[:, None, :] + d[:, None, :] * ts[None, :, None]

    q = q.reshape(-1, 3)
    occu = query_occupancy(p, n, a, q)
    occu = occu.reshape(-1, tsamples)
    
    occu_1 = occu[..., :-1]
    occu_2 = occu[..., 1:]

    # alpha = 1 - (1 - np.maximum(occu_1, occu_2)) / (1 - np.minimum(occu_1, occu_2) + 1e-8)
    alpha = 1 - np.clip((1 - occu_2) / (1 - occu_1 + 1e-8), 0, 1)
    tr = np.cumprod(1 - alpha, axis=-1)

    alpha = np.insert(alpha, -1, 0., axis=-1)
    tr = np.insert(tr, 0, 1., axis=-1)
    ffd = tr * alpha

    # alpha = a1, a2, a3, ..., 0
    # tr = 1 - a1, (1 - a1) * (1 - a2), ...
    # ffd = a1, (1 - a1) * a2, (1 - a1) * (1 - a2) * a3, ...

    return alpha, tr, ffd


def visualize_ffd(alpha, tr, ffd, h, w):
    depths = np.linspace(0, 1, ffd.shape[-1])
    depth_image = ((1 - depths[None, :]) * ffd).sum(axis=-1).reshape(h, w)

    ffd_sum = ffd.sum(axis=-1)
    ffd_normalized = np.insert(ffd, -1, 1 - ffd_sum, axis=-1)

    entropy = (-(ffd_normalized * np.log(ffd_normalized + 1e-8))).sum(axis=-1)
    entropy_image = entropy.reshape(h, w)

    entropy_mask = entropy > np.percentile(entropy, 90)
    entropy_masked_image = (entropy * entropy_mask).reshape(h, w)

    # max_var_index = entropy[entropy_mask].argmax()
    # plt.plot(range(ffd_normalized.shape[-1]), ffd_normalized[entropy_mask][max_var_index], label='ff-distribution')
    # plt.plot(range(alpha.shape[-1]), alpha[entropy_mask][max_var_index], label='opacity')
    # plt.plot(range(tr.shape[-1]), tr[entropy_mask][max_var_index], label='transmittance')
    # plt.legend()
    # plt.show()
    # min_var_index = entropy[entropy_mask].argmin()
    # plt.plot(range(ffd_normalized.shape[-1]), ffd_normalized[entropy_mask][min_var_index], label='ff-distribution')
    # plt.plot(range(alpha.shape[-1]), alpha[entropy_mask][min_var_index], label='opacity')
    # plt.plot(range(tr.shape[-1]), tr[entropy_mask][min_var_index], label='transmittance')
    # plt.legend()
    # plt.show()

    plt.figure(figsize=(5,5))
    plt.imshow(depth_image.transpose(1, 0)[::-1], vmin=0.2, vmax=0.5, cmap='bone')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    plt.imshow(entropy_masked_image.transpose(1, 0)[::-1], vmin=0, vmax=3.5, cmap='magma')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def cluster_ray_directions(d_candidates, m):
    kmeans = KMeans(n_clusters=m, n_init='auto')
    kmeans.fit(d_candidates)
    centroids = kmeans.cluster_centers_
    return centroids / (np.linalg.norm(centroids, axis=-1)[:, None] + 1e-8)


def reconstruct_mesh(p, n):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.normals = o3d.utility.Vector3dVector(n)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    return mesh
