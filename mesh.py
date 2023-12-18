import numpy as np
import torch
import trimesh        
from trimesh.transformations import scale_matrix, translation_matrix
from chamferdist import ChamferDistance


class Mesh:
    def __init__(self, path):
        self.mesh : trimesh.Trimesh = trimesh.load_mesh(path)
        center = (self.mesh.bounds[0] + self.mesh.bounds[1]) / 2
        self.mesh.apply_translation(-center)
        scale = scale_matrix(1.0 / max(self.mesh.extents))
        self.mesh.apply_transform(scale)

    def sample_points(self, num_samples):
        samples, face_indices = trimesh.sample.sample_surface(self.mesh, num_samples)
        normals = self.mesh.face_normals[face_indices]
        return samples, normals

    def intersect_rays(self, o, d):
        locations_ray, index_ray, index_tri = self.mesh.ray.intersects_location(o.reshape(-1, 3), d.reshape(-1, 3), multiple_hits=False)

        hit = np.zeros(len(o), dtype=np.int64)
        hit[index_ray] = 1

        locations = np.zeros((len(o), 3), dtype=np.float64)
        locations[index_ray] = locations_ray

        normals = np.zeros((len(o), 3), dtype=np.float64)
        normals[index_ray] = self.mesh.face_normals[index_tri]

        return hit, locations, normals
    
    def compute_chamfer_distance(self, path, num_samples):
        mesh_other : trimesh.Trimesh = trimesh.load_mesh(path)
        cd = ChamferDistance()
        points_self, _ = trimesh.sample.sample_surface(self.mesh, num_samples)
        points_other, _ = trimesh.sample.sample_surface(mesh_other, num_samples)
        with torch.no_grad():
            dist = cd.forward(
                torch.tensor(points_other, dtype=torch.float32, device='cuda')[None, ...], 
                torch.tensor(points_self, dtype=torch.float32, device='cuda')[None, ...],
                bidirectional=True,
                point_reduction='mean')
        return dist.cpu().item()



class Cube(Mesh):
    def __init__(self, side_length):
        self.mesh = trimesh.creation.box(extents=(side_length, side_length, side_length))
        self.mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [0, 0, 1]))
        self.mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [1, 0, 0]))


class Cone(Mesh):
    def __init__(self, radius, height):
        self.mesh = trimesh.creation.cone(radius, height)

        self.mesh.apply_translation((0, 0, -height/2))

        scale = scale_matrix(1.0 / max(self.mesh.extents))
        self.mesh.apply_transform(scale)
