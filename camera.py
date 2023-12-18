from scipy.spatial.transform import Rotation as R
import numpy as np

# from utils import visualize_points
# import matplotlib.pyplot as plt


class Camera:
    def __init__(self, axis='x', angle=90):
        self.rotation = R.from_euler(axis, angle, degrees=True)
        self.base_camera = np.array([0., 0., 1.5])
        self.base_image_plane = np.array([0., 0., 1.1])
        self.base_image_plane_size = (0.5, 0.5)

    def sample_rays(self, num_rays=(4, 4)):
        pos_xy = np.stack(np.meshgrid(
            np.linspace(-self.base_image_plane_size[0] / 2, self.base_image_plane_size[0] / 2, num_rays[0], endpoint=False),
            np.linspace(-self.base_image_plane_size[1] / 2, self.base_image_plane_size[1] / 2, num_rays[1], endpoint=False),
            indexing='ij'
        ), axis=-1).reshape(-1, 2)
        pos_xy[:, 0] += self.base_image_plane_size[0] / num_rays[0] / 2
        pos_xy[:, 1] += self.base_image_plane_size[1] / num_rays[1] / 2

        pos_z = np.zeros((pos_xy.shape[0], 1))
        pos_xyz = np.concatenate((pos_xy, pos_z), axis=-1)

        pos_xyz += self.base_image_plane

        pos_rot = self.rotation.apply(pos_xyz)
        camera_rot = self.rotation.apply(self.base_camera)

        origins = camera_rot[None, :].repeat(pos_rot.shape[0], 0)
        directions = pos_rot - origins
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        return origins, directions
