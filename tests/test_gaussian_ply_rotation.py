"""Run with ComfyUI's Python environment; no model weights are needed."""
import ast
import os
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

# Load the actual class without importing unrelated CUDA sparse/model backends.
NODE_ROOT = Path(os.environ.get('SAM3D_NODE_ROOT', Path(__file__).resolve().parents[1]))
source = Path(os.environ.get('SAM3D_SOURCE', NODE_ROOT / 'nodes/sam3d/representations.py'))
tree = ast.parse(source.read_text())
nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
         and n.name in {'Gaussian', 'inverse_sigmoid'}]
scope = dict(np=np, torch=torch, PlyData=PlyData, PlyElement=PlyElement, Rotation=Rotation,
             comfy=types.SimpleNamespace(model_management=types.SimpleNamespace(
                 get_torch_device=lambda: torch.device('cpu'))))
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
Gaussian = scope['Gaussian']


class GaussianPlyRotationTest(unittest.TestCase):
    def setUp(self):
        self.g = Gaussian(aabb=[0, 0, 0, 1, 1, 1], device='cpu')
        self.g._xyz = torch.tensor([[0.2, -0.3, 0.6], [-0.1, 0.7, 0.4]])
        self.g._features_dc = torch.tensor([[[0.1, 0.2, 0.3]], [[-0.2, 0.5, 0.1]]])
        self.g._opacity = torch.tensor([[0.4], [-0.7]])
        self.g._scaling = torch.log(torch.tensor([[1., 3., 7.], [5., 2., 1.]]))
        self.g._rotation = torch.tensor([[1., 0., 0., 0.], [1., 2., -3., 4.]]) - self.g.rots_bias
        self.axis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def save(self, name, transform=None):
        p = Path(self.tmp.name) / name
        self.g.save_ply(str(p), transform=transform)
        return PlyData.read(p)['vertex'].data.copy()

    @staticmethod
    def vectors(data, fields):
        return np.stack([data[f] for f in fields], axis=-1)

    def covariance(self, data):
        q = self.vectors(data, ['rot_0', 'rot_1', 'rot_2', 'rot_3'])
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q.T
        r = np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                      2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                      2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], axis=1).reshape(-1,3,3)
        scale = np.exp(self.vectors(data, ['scale_0', 'scale_1', 'scale_2']))
        m = r * scale[:, None, :]
        return m @ m.transpose(0, 2, 1)

    def test_y_up_rotates_centers_and_anisotropic_covariance(self):
        before = self.save('before.ply')
        after = self.save('after.ply', self.axis)
        np.testing.assert_allclose(self.vectors(after, ['x', 'y', 'z']),
                                   self.vectors(before, ['x', 'y', 'z']) @ self.axis)
        np.testing.assert_allclose(self.covariance(after),
                                   self.axis.T @ self.covariance(before) @ self.axis,
                                   atol=1e-8, rtol=2e-6)
        for field in before.dtype.names:
            if field not in ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3']:
                np.testing.assert_array_equal(after[field], before[field])

    def test_no_transform_preserves_raw_rotation_and_model_state(self):
        state = {k: v.clone() for k, v in vars(self.g).items() if isinstance(v, torch.Tensor)}
        before = self.save('none.ply')
        np.testing.assert_array_equal(self.vectors(before, ['rot_0', 'rot_1', 'rot_2', 'rot_3']),
                                      (self.g._rotation + self.g.rots_bias).numpy())
        self.save('rotated.ply', self.axis)
        for key, value in state.items():
            torch.testing.assert_close(vars(self.g)[key], value, rtol=0, atol=0)
        after = self.save('none_again.ply')
        np.testing.assert_array_equal(before, after)

    def test_inverse_rotation_restores_covariance(self):
        before = self.save('before.ply')
        after = self.save('rotated.ply', self.axis)
        self.g._xyz = torch.from_numpy(self.vectors(after, ['x', 'y', 'z']))
        self.g._rotation = torch.from_numpy(self.vectors(after, ['rot_0', 'rot_1', 'rot_2', 'rot_3'])) - self.g.rots_bias
        restored = self.save('restored.ply', self.axis.T)
        np.testing.assert_allclose(self.vectors(restored, ['x', 'y', 'z']), self.vectors(before, ['x', 'y', 'z']))
        np.testing.assert_allclose(self.covariance(restored), self.covariance(before), atol=1e-8, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
