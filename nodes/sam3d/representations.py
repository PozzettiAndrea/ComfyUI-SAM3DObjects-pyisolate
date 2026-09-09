# Copyright (c) Meta Platforms, Inc. and affiliates.
# Consolidated from tdfy_dit/renderers/sh_utils.py, representations/gaussian/,
# representations/mesh/, representations/octree/, representations/radiance_field/
"""
3D representations: Gaussian, MeshExtract, FlexiCubes, DfsOctree, Strivec.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from datetime import datetime
import random
import comfy.model_management
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
from easydict import EasyDict as edict
from .sparse import SparseTensor

log = logging.getLogger("sam3dobjects")


# =============================================================================
# SH utilities (from renderers/sh_utils.py)
# =============================================================================

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


# =============================================================================
# Gaussian general utilities (from representations/gaussian/general_utils.py)
# =============================================================================

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


# =============================================================================
# Gaussian model (from representations/gaussian/gaussian_model.py)
# =============================================================================

log = logging.getLogger("sam3dobjects")


class Gaussian:
    def __init__(
        self,
        aabb: list,
        sh_degree: int = 0,
        mininum_kernel_size: float = 0.0,
        scaling_bias: float = 0.01,
        opacity_bias: float = 0.1,
        scaling_activation: str = "exp",
        device="cuda",
    ):
        self.init_params = {
            "aabb": aabb,
            "sh_degree": sh_degree,
            "mininum_kernel_size": mininum_kernel_size,
            "scaling_bias": scaling_bias,
            "opacity_bias": opacity_bias,
            "scaling_activation": scaling_activation,
        }

        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    def setup_functions(self):
        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = softplus_inverse_scaling_activation

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        device = comfy.model_management.get_torch_device()
        self.scale_bias = self.inverse_scaling_activation(
            torch.tensor(self.scaling_bias)
        ).to(device)
        self.rots_bias = torch.zeros((4)).to(device)
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(
            torch.tensor(self.opacity_bias)
        ).to(device)

    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size**2
        scales = torch.sqrt(scales)
        return scales

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])

    @property
    def get_xyz(self):
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]

    @property
    def get_features(self):
        return (
            torch.cat((self._features_dc, self._features_rest), dim=2)
            if self._features_rest is not None
            else self._features_dc
        )

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity + self.opacity_bias)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :]
        )

    def from_scaling(self, scales):
        scales = torch.sqrt(torch.square(scales) - self.mininum_kernel_size**2)
        self._scaling = self.inverse_scaling_activation(scales) - self.scale_bias

    def from_rotation(self, rots):
        self._rotation = rots - self.rots_bias[None, :]

    def from_xyz(self, xyz):
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]

    def from_features(self, features):
        self._features_dc = features

    def from_opacity(self, opacities):
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path, transform=None):
        xyz = self.get_xyz.detach().cpu().numpy()
        if transform is not None:
            xyz = xyz @ transform
        normals = np.zeros_like(xyz)

        # Prepare raw SH coefficients (for Gaussian Splatting viewers)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )

        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        if opacities.ndim == 1:
            opacities = opacities[:, np.newaxis]

        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()
        if transform is not None:
            # xyz uses row vectors; Gaussian rotations use column vectors.
            rotation = (
                Rotation.from_matrix(transform.T)
                * Rotation.from_quat(rotation[:, [1, 2, 3, 0]])
            ).as_quat()[:, [3, 0, 1, 2]]

        # Standard Gaussian Splatting PLY format (float-only for gsplat.js compat)
        dtype_full = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                      ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]

        for i in range(f_dc.shape[1]):
            dtype_full.append(("f_dc_{}".format(i), "f4"))

        dtype_full.append(("opacity", "f4"))

        for i in range(scale.shape[1]):
            dtype_full.append(("scale_{}".format(i), "f4"))
        for i in range(rotation.shape[1]):
            dtype_full.append(("rot_{}".format(i), "f4"))

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")

        # Write as ASCII format for VTK.js compatibility
        plydata = PlyData([el])
        plydata.text = True
        plydata.write(path)

    def load_ply(self, path):
        log.info("load_ply: Loading from %s", path)
        plydata = PlyData.read(path)
        log.info("load_ply: PLY loaded, %d vertices", len(plydata.elements[0]))
        log.debug("load_ply: Properties: %s", [p.name for p in plydata.elements[0].properties])

        log.debug("load_ply: Extracting xyz coordinates...")
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        log.debug("load_ply: xyz shape=%s, dtype=%s, strides=%s", xyz.shape, xyz.dtype, xyz.strides)

        log.debug("load_ply: Extracting opacities...")
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].copy()
        log.debug("load_ply: opacities shape=%s, dtype=%s, strides=%s", opacities.shape, opacities.dtype, opacities.strides)

        log.debug("load_ply: Extracting features_dc...")
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        log.debug("load_ply: features_dc shape=%s, dtype=%s", features_dc.shape, features_dc.dtype)

        if self.sh_degree > 0:
            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
            )

        log.debug("load_ply: Extracting scales...")
        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        log.debug("load_ply: scales shape=%s, dtype=%s", scales.shape, scales.dtype)

        log.debug("load_ply: Extracting rotations...")
        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        log.debug("load_ply: rots shape=%s, dtype=%s", rots.shape, rots.dtype)

        # convert to actual gaussian attributes
        log.debug("load_ply: Converting to torch tensors on device=%s...", self.device)
        log.debug("load_ply: Converting xyz...")
        xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        log.debug("load_ply: Converting features_dc...")
        features_dc = (
            torch.tensor(features_dc, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
        )
        if self.sh_degree > 0:
            log.debug("load_ply: Converting features_extra...")
            features_extra = (
                torch.tensor(features_extra, dtype=torch.float, device=self.device)
                .transpose(1, 2)
                .contiguous()
            )
        log.debug("load_ply: Converting opacities...")
        opacities = torch.sigmoid(
            torch.tensor(opacities, dtype=torch.float, device=self.device)
        )
        log.debug("load_ply: Converting scales...")
        scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device))
        log.debug("load_ply: Converting rots...")
        rots = torch.tensor(rots, dtype=torch.float, device=self.device)

        # convert to _hidden attributes
        log.debug("load_ply: Setting internal attributes...")
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        self._features_dc = features_dc
        if self.sh_degree > 0:
            self._features_rest = features_extra
        else:
            self._features_rest = None
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias
        self._scaling = (
            self.inverse_scaling_activation(
                torch.sqrt(torch.square(scales) - self.mininum_kernel_size**2)
            )
            - self.scale_bias
        )
        self._rotation = rots - self.rots_bias[None, :]
        log.info("load_ply: Complete! Loaded %d gaussians", self._xyz.shape[0])

def softplus_inverse_scaling_activation(x):
    return x + torch.log(-torch.expm1(-x))

# =============================================================================
# FlexiCubes lookup tables (from representations/mesh/flexicubes/tables.py)
# =============================================================================

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
dmc_table = [
[[-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 8, 9, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 7, 8, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 7, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [4, 7, 8, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 4, 7, 9, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 9, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [4, 5, 9, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 5, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 4, 5, 8, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[5, 7, 8, 9, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 5, 7, 9, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 5, 7, 8, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 5, 7, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 8, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 8, 9, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 7, 8, -1, -1, -1, -1], [2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 4, 7, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [4, 7, 8, -1, -1, -1, -1], [2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 4, 7, 9, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 9, -1, -1, -1, -1], [2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 8, 11, -1, -1, -1], [4, 5, 9, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 5, -1, -1, -1], [2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 4, 5, 8, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[5, 7, 8, 9, -1, -1, -1], [2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 5, 7, 9, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 5, 7, 8, -1, -1], [2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 5, 7, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 9, 10, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 8, 9, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 7, 8, -1, -1, -1, -1], [1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 7, -1, -1, -1], [1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 9, 10, -1, -1, -1], [4, 7, 8, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 4, 7, 9, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 9, -1, -1, -1, -1], [1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [4, 5, 9, -1, -1, -1, -1], [1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 4, 5, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 4, 5, 8, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[5, 7, 8, 9, -1, -1, -1], [1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 5, 7, 9, -1, -1], [1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 5, 7, 8, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 5, 7, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 10, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 8, 10, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 9, 10, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[8, 9, 10, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 7, 8, -1, -1, -1, -1], [1, 3, 10, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 7, 10, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 9, 10, 11, -1, -1], [4, 7, 8, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 7, 9, 10, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 9, -1, -1, -1, -1], [1, 3, 10, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 8, 10, 11, -1, -1], [4, 5, 9, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 5, 10, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 8, 10, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[5, 7, 8, 9, -1, -1, -1], [1, 3, 10, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 5, 7, 9, 10, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 5, 7, 8, 10, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[5, 7, 10, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 8, 9, -1, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 6, 8, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 6, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [4, 6, 8, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 4, 6, 9, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 9, -1, -1, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [4, 5, 9, -1, -1, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 5, -1, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 4, 5, 8, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[5, 6, 8, 9, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 5, 6, 9, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 5, 6, 8, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 5, 6, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 6, 7, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 6, 7, 8, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [2, 3, 6, 7, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 6, 7, 8, 9, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 4, 6, 8, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 4, 6, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [2, 3, 4, 6, 8, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 4, 6, 9, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 9, -1, -1, -1, -1], [2, 3, 6, 7, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 6, 7, 8, -1, -1], [4, 5, 9, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 5, -1, -1, -1], [2, 3, 6, 7, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 4, 5, 6, 7, 8], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 5, 6, 8, 9, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 5, 6, 9, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 2, 3, 5, 6, 8], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 5, 6, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 10, -1, -1, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [1, 2, 10, -1, -1, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 9, 10, -1, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 8, 9, 10, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 6, 8, 11, -1, -1, -1], [1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 6, 11, -1, -1], [1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 9, 10, -1, -1, -1], [4, 6, 8, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 4, 6, 9, 10, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 9, -1, -1, -1, -1], [1, 2, 10, -1, -1, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [4, 5, 9, -1, -1, -1, -1], [1, 2, 10, -1, -1, -1, -1], [6, 7, 11, -1, -1, -1, -1]],
[[0, 2, 4, 5, 10, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 4, 5, 8, 10, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[5, 6, 8, 9, 11, -1, -1], [1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 5, 6, 9, 11, -1], [1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 5, 6, 8, 10, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 5, 6, 10, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 6, 7, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 6, 7, 8, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 6, 7, 9, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[6, 7, 8, 9, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 4, 6, 8, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 6, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 6, 8, 9, 10], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 6, 9, 10, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 9, -1, -1, -1, -1], [1, 3, 6, 7, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 6, 7, 8, 10, -1], [4, 5, 9, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 5, 6, 7, 10], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 6, 7, 8, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 5, 6, 8, 9, 10], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 5, 6, 9, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 8, 9, -1, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 7, 8, -1, -1, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 7, -1, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [4, 7, 8, -1, -1, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 4, 7, 9, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 6, 9, 10, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [4, 6, 9, 10, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 6, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 4, 6, 8, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[6, 7, 8, 9, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 6, 7, 9, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 6, 7, 8, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 6, 7, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 11, -1, -1, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 8, 11, -1, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [2, 3, 11, -1, -1, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 8, 9, 11, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 7, 8, -1, -1, -1, -1], [2, 3, 11, -1, -1, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 4, 7, 11, -1, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [4, 7, 8, -1, -1, -1, -1], [2, 3, 11, -1, -1, -1, -1], [5, 6, 10, -1, -1, -1, -1]],
[[1, 2, 4, 7, 9, 11, -1], [5, 6, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 6, 9, 10, -1, -1, -1], [2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 8, 11, -1, -1, -1], [4, 6, 9, 10, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 6, 10, -1, -1], [2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 4, 6, 8, 10, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[6, 7, 8, 9, 10, -1, -1], [2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 6, 7, 9, 10, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 6, 7, 8, 10, -1], [2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 6, 7, 10, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 5, 6, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [1, 2, 5, 6, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 5, 6, 9, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 5, 6, 8, 9, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 7, 8, -1, -1, -1, -1], [1, 2, 5, 6, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 7, -1, -1, -1], [1, 2, 5, 6, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 5, 6, 9, -1, -1], [4, 7, 8, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 4, 5, 6, 7, 9], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 4, 6, 9, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [1, 2, 4, 6, 9, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 4, 6, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 4, 6, 8, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 6, 7, 8, 9, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 2, 3, 6, 7, 9], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 6, 7, 8, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 6, 7, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 5, 6, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 5, 6, 8, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 5, 6, 9, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[5, 6, 8, 9, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 7, 8, -1, -1, -1, -1], [1, 3, 5, 6, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 5, 6, 7, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 5, 6, 9, 11, -1], [4, 7, 8, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 6, 7, 9, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 4, 6, 9, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 6, 8, 9, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 6, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 6, 8, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 6, 7, 8, 9, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 6, 7, 8, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[6, 7, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[5, 7, 10, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [5, 7, 10, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [5, 7, 10, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 8, 9, -1, -1, -1], [5, 7, 10, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 8, 10, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 5, 10, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [4, 5, 8, 10, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 4, 5, 9, 10, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 7, 9, 10, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [4, 7, 9, 10, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 7, 10, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 4, 7, 8, 10, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[8, 9, 10, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 9, 10, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 8, 10, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 10, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 5, 7, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 5, 7, 8, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [2, 3, 5, 7, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 5, 7, 8, 9, 10], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 4, 5, 8, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 4, 5, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [2, 3, 4, 5, 8, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 4, 5, 9, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 4, 7, 9, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 4, 7, 8, 9, 10], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 2, 3, 4, 7, 10], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 7, 8, -1, -1, -1, -1], [1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 8, 9, 10, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 9, 10, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 2, 3, 8, 10, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 10, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 5, 7, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [1, 2, 5, 7, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 5, 7, 9, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 5, 7, 8, 9, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 4, 5, 8, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 2, 3, 4, 5, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 4, 5, 8, 9, 11], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 9, -1, -1, -1, -1], [2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 4, 7, 9, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [1, 2, 4, 7, 9, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 4, 7, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 4, 7, 8, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 2, 8, 9, 11, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 2, 3, 9, 11, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 2, 8, 11, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[2, 3, 11, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 5, 7, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 5, 7, 8, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 5, 7, 9, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[5, 7, 8, 9, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 4, 5, 8, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 5, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 5, 8, 9, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 5, 9, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 4, 7, 9, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 4, 7, 8, 9, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 4, 7, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[4, 7, 8, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[1, 3, 8, 9, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 1, 9, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[0, 3, 8, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
[[-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]]
]
num_vd_table = [0, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 3, 1, 2, 2,
2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 2, 3, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2,
1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 3, 2, 2, 1, 1, 1, 1,
1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 2, 2, 1, 3, 4, 2,
2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2,
3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 3, 2, 4, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1,
2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1,
1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2,
1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
check_table = [
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 1, 0, 0, 194],
[1, -1, 0, 0, 193],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, 1, 0, 164],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, -1, 0, 161],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, 0, 1, 152],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, 0, 1, 145],
[1, 0, 0, 1, 144],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, 0, -1, 137],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, 1, 0, 133],
[1, 0, 1, 0, 132],
[1, 1, 0, 0, 131],
[1, 1, 0, 0, 130],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, 0, 1, 100],
[0, 0, 0, 0, 0],
[1, 0, 0, 1, 98],
[0, 0, 0, 0, 0],
[1, 0, 0, 1, 96],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, 1, 0, 88],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, -1, 0, 82],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, 1, 0, 74],
[0, 0, 0, 0, 0],
[1, 0, 1, 0, 72],
[0, 0, 0, 0, 0],
[1, 0, 0, -1, 70],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, -1, 0, 0, 67],
[0, 0, 0, 0, 0],
[1, -1, 0, 0, 65],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 1, 0, 0, 56],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, -1, 0, 0, 52],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 1, 0, 0, 44],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 1, 0, 0, 40],
[0, 0, 0, 0, 0],
[1, 0, 0, -1, 38],
[1, 0, -1, 0, 37],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, -1, 0, 33],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, -1, 0, 0, 28],
[0, 0, 0, 0, 0],
[1, 0, -1, 0, 26],
[1, 0, 0, -1, 25],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, -1, 0, 0, 20],
[0, 0, 0, 0, 0],
[1, 0, -1, 0, 18],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, 0, -1, 9],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[1, 0, 0, -1, 6],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0]
]
tet_table = [
[-1, -1, -1, -1, -1, -1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1],
[4, 4, 4, 4, 4, 4],
[0, 0, 0, 0, 0, 0],
[4, 0, 0, 4, 4, -1],
[1, 1, 1, 1, 1, 1],
[4, 4, 4, 4, 4, 4],
[0, 4, 0, 4, 4, -1],
[0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1],
[5, 5, 5, 5, 5, 5],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1],
[2, 2, 2, 2, 2, 2],
[0, 0, 0, 0, 0, 0],
[2, 0, 2, -1, 0, 2],
[1, 1, 1, 1, 1, 1],
[2, -1, 2, 4, 4, 2],
[0, 0, 0, 0, 0, 0],
[2, 0, 2, 4, 4, 2],
[1, 1, 1, 1, 1, 1],
[2, 4, 2, 4, 4, 2],
[0, 4, 0, 4, 4, 0],
[2, 0, 2, 0, 0, 2],
[1, 1, 1, 1, 1, 1],
[2, 5, 2, 5, 5, 2],
[0, 0, 0, 0, 0, 0],
[2, 0, 2, 0, 0, 2],
[1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1],
[0, 1, 1, -1, 0, 1],
[0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 2],
[4, 1, 1, 4, 4, 1],
[0, 1, 1, 0, 0, 1],
[4, 0, 0, 4, 4, 0],
[2, 2, 2, 2, 2, 2],
[-1, 1, 1, 4, 4, 1],
[0, 1, 1, 4, 4, 1],
[0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 2],
[5, 1, 1, 5, 5, 1],
[0, 1, 1, 0, 0, 1],
[0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[8, 8, 8, 8, 8, 8],
[1, 1, 1, 4, 4, 1],
[0, 0, 0, 0, 0, 0],
[4, 0, 0, 4, 4, 0],
[4, 4, 4, 4, 4, 4],
[1, 1, 1, 4, 4, 1],
[0, 4, 0, 4, 4, 0],
[0, 0, 0, 0, 0, 0],
[4, 4, 4, 4, 4, 4],
[1, 1, 1, 5, 5, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[5, 5, 5, 5, 5, 5],
[6, 6, 6, 6, 6, 6],
[6, -1, 0, 6, 0, 6],
[6, 0, 0, 6, 0, 6],
[6, 1, 1, 6, 1, 6],
[4, 4, 4, 4, 4, 4],
[0, 0, 0, 0, 0, 0],
[4, 0, 0, 4, 4, 4],
[1, 1, 1, 1, 1, 1],
[6, 4, -1, 6, 4, 6],
[6, 4, 0, 6, 4, 6],
[6, 0, 0, 6, 0, 6],
[6, 1, 1, 6, 1, 6],
[5, 5, 5, 5, 5, 5],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1],
[2, 2, 2, 2, 2, 2],
[0, 0, 0, 0, 0, 0],
[2, 0, 2, 2, 0, 2],
[1, 1, 1, 1, 1, 1],
[2, 2, 2, 2, 2, 2],
[0, 0, 0, 0, 0, 0],
[2, 0, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1],
[2, 4, 2, 2, 4, 2],
[0, 4, 0, 4, 4, 0],
[2, 0, 2, 2, 0, 2],
[1, 1, 1, 1, 1, 1],
[2, 2, 2, 2, 2, 2],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1],
[6, 1, 1, 6, -1, 6],
[6, 1, 1, 6, 0, 6],
[6, 0, 0, 6, 0, 6],
[6, 2, 2, 6, 2, 6],
[4, 1, 1, 4, 4, 1],
[0, 1, 1, 0, 0, 1],
[4, 0, 0, 4, 4, 4],
[2, 2, 2, 2, 2, 2],
[6, 1, 1, 6, 4, 6],
[6, 1, 1, 6, 4, 6],
[6, 0, 0, 6, 0, 6],
[6, 2, 2, 6, 2, 6],
[5, 1, 1, 5, 5, 1],
[0, 1, 1, 0, 0, 1],
[0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[6, 6, 6, 6, 6, 6],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[4, 4, 4, 4, 4, 4],
[1, 1, 1, 1, 4, 1],
[0, 4, 0, 4, 4, 0],
[0, 0, 0, 0, 0, 0],
[4, 4, 4, 4, 4, 4],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 5, 0, 5, 0, 5],
[5, 5, 5, 5, 5, 5],
[5, 5, 5, 5, 5, 5],
[0, 5, 0, 5, 0, 5],
[-1, 5, 0, 5, 0, 5],
[1, 5, 1, 5, 1, 5],
[4, 5, -1, 5, 4, 5],
[0, 5, 0, 5, 0, 5],
[4, 5, 0, 5, 4, 5],
[1, 5, 1, 5, 1, 5],
[4, 4, 4, 4, 4, 4],
[0, 4, 0, 4, 4, 4],
[0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1],
[6, 6, 6, 6, 6, 6],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1],
[2, 5, 2, 5, -1, 5],
[0, 5, 0, 5, 0, 5],
[2, 5, 2, 5, 0, 5],
[1, 5, 1, 5, 1, 5],
[2, 5, 2, 5, 4, 5],
[0, 5, 0, 5, 0, 5],
[2, 5, 2, 5, 4, 5],
[1, 5, 1, 5, 1, 5],
[2, 4, 2, 4, 4, 2],
[0, 4, 0, 4, 4, 4],
[2, 0, 2, 0, 0, 2],
[1, 1, 1, 1, 1, 1],
[2, 6, 2, 6, 6, 2],
[0, 0, 0, 0, 0, 0],
[2, 0, 2, 0, 0, 2],
[1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1],
[0, 1, 1, 1, 0, 1],
[0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 2],
[4, 1, 1, 1, 4, 1],
[0, 1, 1, 1, 0, 1],
[4, 0, 0, 4, 4, 0],
[2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1],
[0, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[5, 5, 5, 5, 5, 5],
[1, 1, 1, 1, 4, 1],
[0, 0, 0, 0, 0, 0],
[4, 0, 0, 4, 4, 0],
[4, 4, 4, 4, 4, 4],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[4, 4, 4, 4, 4, 4],
[1, 1, 1, 1, 1, 1],
[6, 0, 0, 6, 0, 6],
[0, 0, 0, 0, 0, 0],
[6, 6, 6, 6, 6, 6],
[5, 5, 5, 5, 5, 5],
[5, 5, 0, 5, 0, 5],
[5, 5, 0, 5, 0, 5],
[5, 5, 1, 5, 1, 5],
[4, 4, 4, 4, 4, 4],
[0, 0, 0, 0, 0, 0],
[4, 4, 0, 4, 4, 4],
[1, 1, 1, 1, 1, 1],
[4, 4, 4, 4, 4, 4],
[4, 4, 0, 4, 4, 4],
[0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1],
[8, 8, 8, 8, 8, 8],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1],
[2, 2, 2, 2, 2, 2],
[0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 0, 2],
[1, 1, 1, 1, 1, 1],
[2, 2, 2, 2, 2, 2],
[0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1],
[2, 2, 2, 2, 2, 2],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[4, 1, 1, 4, 4, 1],
[2, 2, 2, 2, 2, 2],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 0, 1],
[0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[2, 4, 2, 4, 4, 2],
[1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 2],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[5, 5, 5, 5, 5, 5],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[4, 4, 4, 4, 4, 4],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[4, 4, 4, 4, 4, 4],
[1, 1, 1, 1, 1, 1],
[0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0],
[12, 12, 12, 12, 12, 12]
]


# =============================================================================
# FlexiCubes (from representations/mesh/flexicubes/flexicubes.py)
# =============================================================================

class FlexiCubes:
    def __init__(self, device="cuda"):

        self.device = device
        self.dmc_table = torch.tensor(dmc_table, dtype=torch.int32, device=device, requires_grad=False)
        self.num_vd_table = torch.tensor(num_vd_table,
                                         dtype=torch.int32, device=device, requires_grad=False)
        self.check_table = torch.tensor(
            check_table,
            dtype=torch.int32, device=device, requires_grad=False)

        self.tet_table = torch.tensor(tet_table, dtype=torch.int32, device=device, requires_grad=False)
        self.quad_split_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.int32, device=device, requires_grad=False)
        self.quad_split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.int32, device=device, requires_grad=False)

        self.cube_corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [
                                         1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float, device=device)
        self.cube_corners_idx = torch.pow(2, torch.arange(8, dtype=torch.int32, requires_grad=False))
        self.cube_edges = torch.tensor([0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6,
                                       2, 0, 3, 1, 7, 5, 6, 4], dtype=torch.int32, device=device, requires_grad=False)

        self.edge_dir_table = torch.tensor([0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1],
                                           dtype=torch.int32, device=device)
        self.dir_faces_table = torch.tensor([
            [[5, 4], [3, 2], [4, 5], [2, 3]],
            [[5, 4], [1, 0], [4, 5], [0, 1]],
            [[3, 2], [1, 0], [2, 3], [0, 1]]
        ], dtype=torch.int32, device=device)
        self.adj_pairs = torch.tensor([0, 1, 1, 3, 3, 2, 2, 0], dtype=torch.int32, device=device)

    def __call__(self, voxelgrid_vertices, scalar_field, cube_idx, resolution, qef_reg_scale=1e-3,
                 weight_scale=0.99, beta=None, alpha=None, gamma_f=None, voxelgrid_colors=None,
                 cube_index_map=None, dtype=None):  # cube_index_map required for sparse path
        """
        Optionally specify 'dtype' to unify all float inputs (e.g. torch.float16).
        If 'dtype' is None, we infer from voxelgrid_vertices or default to float32.
        """
        # unify floating tensors to a chosen dtype, so it can work with half, etc:
        if dtype is None:
            if voxelgrid_vertices.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                dtype = voxelgrid_vertices.dtype
            else:
                dtype = torch.float32

        voxelgrid_vertices = voxelgrid_vertices.to(dtype)
        scalar_field = scalar_field.to(dtype)
        if beta is not None:
            beta = beta.to(dtype)
        if alpha is not None:
            alpha = alpha.to(dtype)
        if gamma_f is not None:
            gamma_f = gamma_f.to(dtype)
        if voxelgrid_colors is not None:
            voxelgrid_colors = voxelgrid_colors.to(dtype)

        import logging
        _fclog = logging.getLogger("sam3dobjects")
        def _fvm():
            return torch.cuda.memory_allocated() / 1024**2
        _fclog.warning("[VRAM:fc] start: %.0f MiB | verts %s cubes %s sdf %s", _fvm(), voxelgrid_vertices.shape, cube_idx.shape, scalar_field.shape)

        surf_cubes, occ_fx8 = self._identify_surf_cubes(scalar_field, cube_idx)
        if surf_cubes.sum() == 0:
            return (
                torch.zeros((0, 3), device=self.device, dtype=dtype),
                torch.zeros((0, 3), dtype=torch.int32, device=self.device),
                torch.zeros((0), device=self.device, dtype=dtype),
                torch.zeros((0, voxelgrid_colors.shape[-1]), device=self.device, dtype=dtype) if voxelgrid_colors is not None else None
            )
        _fclog.warning("[VRAM:fc] after _identify_surf_cubes: %.0f MiB | surf_cubes %d/%d", _fvm(), surf_cubes.sum().item(), surf_cubes.shape[0])

        beta, alpha, gamma_f = self._normalize_weights(
            beta, alpha, gamma_f, surf_cubes, weight_scale)

        if voxelgrid_colors is not None:
            voxelgrid_colors = torch.sigmoid(voxelgrid_colors)

        case_ids = self._get_case_id(occ_fx8, surf_cubes, resolution, cube_index_map)
        _fclog.warning("[VRAM:fc] after _get_case_id: %.0f MiB", _fvm())

        surf_edges, idx_map, edge_counts, surf_edges_mask = self._identify_surf_edges(
            scalar_field, cube_idx, surf_cubes
        )
        _fclog.warning("[VRAM:fc] after _identify_surf_edges: %.0f MiB | surf_edges %s", _fvm(), surf_edges.shape)

        vd, L_dev, vd_gamma, vd_idx_map, vd_color = self._compute_vd(
            voxelgrid_vertices, cube_idx[surf_cubes], surf_edges, scalar_field,
            case_ids, beta, alpha, gamma_f, idx_map, qef_reg_scale, voxelgrid_colors)
        _fclog.warning("[VRAM:fc] after _compute_vd: %.0f MiB | vd %s", _fvm(), vd.shape)

        vertices, faces, s_edges, edge_indices, vertices_color = self._triangulate(
            scalar_field, surf_edges, vd, vd_gamma, edge_counts, idx_map,
            vd_idx_map, surf_edges_mask, vd_color)
        _fclog.warning("[VRAM:fc] after _triangulate: %.0f MiB | verts %s faces %s", _fvm(), vertices.shape, faces.shape)
        return vertices, faces, L_dev, vertices_color

    def _compute_reg_loss(self, vd, ue, edge_group_to_vd, vd_num_edges):
        """
        Regularizer L_dev as in Equation 8
        """
        dist = torch.norm(ue - torch.index_select(input=vd, index=edge_group_to_vd, dim=0), dim=-1)
        mean_l2 = torch.zeros_like(vd[:, 0])
        mean_l2 = (mean_l2).index_add_(0, edge_group_to_vd, dist) / vd_num_edges.squeeze(1).float()
        mad = (dist - torch.index_select(input=mean_l2, index=edge_group_to_vd, dim=0)).abs()
        return mad

    def _normalize_weights(self, beta, alpha, gamma_f, surf_cubes, weight_scale):
        """
        Normalizes the given weights to be non-negative. If input weights are None, it creates and returns a set of weights of ones.
        """
        n_cubes = surf_cubes.shape[0]

        if beta is not None:
            beta = (torch.tanh(beta) * weight_scale + 1)
        else:
            beta = torch.ones((n_cubes, 12), dtype=torch.float, device=self.device)

        if alpha is not None:
            alpha = (torch.tanh(alpha) * weight_scale + 1)
        else:
            alpha = torch.ones((n_cubes, 8), dtype=torch.float, device=self.device)

        if gamma_f is not None:
            gamma_f = torch.sigmoid(gamma_f) * weight_scale + (1 - weight_scale) / 2
        else:
            gamma_f = torch.ones((n_cubes), dtype=torch.float, device=self.device)

        return beta[surf_cubes], alpha[surf_cubes], gamma_f[surf_cubes]

    @torch.no_grad()
    def _get_case_id(self, occ_fx8, surf_cubes, res, cube_index_map):
        """
        Obtains the ID of topology cases based on cell corner occupancy. This function resolves the
        ambiguity in the Dual Marching Cubes (DMC) configurations as described in Section 1.3 of the
        supplementary material. Uses hashmap-based sparse neighbor lookup (SparseFlex).
        """
        occ_int = occ_fx8[surf_cubes].to(torch.int32)
        corners_idx = self.cube_corners_idx.to(self.device).unsqueeze(0)
        case_ids = (occ_int * corners_idx).sum(-1, dtype=torch.int32)

        problem_config = self.check_table.to(self.device)[case_ids]
        to_check = problem_config[..., 0] == 1

        problem_config = problem_config[to_check]
        problem_config_index = cube_index_map[surf_cubes][to_check]

        # Build hashmap from ALL problematic cubes BEFORE within_range filtering.
        # This mirrors the original dense code where problem_config_full is populated
        # with all problematic cubes before any adjacency-based filtering.
        all_problem_config = problem_config
        all_keys = ((problem_config_index[..., 0] * res + problem_config_index[..., 1]) * res + problem_config_index[..., 2]).tolist()
        inverse_cube_index_dict = dict(zip(all_keys, range(len(all_problem_config))))
        all_problem_config_sentinel = torch.cat((all_problem_config, torch.zeros((1, 5), dtype=all_problem_config.dtype, device=all_problem_config.device)))

        vol_idx_problem = cube_index_map[surf_cubes][to_check]
        vol_idx_problem_adj = vol_idx_problem + problem_config[..., 1:4]

        within_range = (
            vol_idx_problem_adj[..., 0] >= 0) & (
            vol_idx_problem_adj[..., 0] < res) & (
            vol_idx_problem_adj[..., 1] >= 0) & (
            vol_idx_problem_adj[..., 1] < res) & (
            vol_idx_problem_adj[..., 2] >= 0) & (
            vol_idx_problem_adj[..., 2] < res)

        vol_idx_problem_adj = vol_idx_problem_adj[within_range]
        problem_config = problem_config[within_range]

        # Look up adjacent positions in the FULL hashmap (not the filtered one)
        inverse_indices = ((vol_idx_problem_adj[..., 0] * res + vol_idx_problem_adj[..., 1]) * res + vol_idx_problem_adj[..., 2]).tolist()
        inverse_indices = torch.tensor([inverse_cube_index_dict.get(idx, len(all_problem_config)) for idx in inverse_indices], dtype=torch.int32, device=self.device)
        problem_config_adj = all_problem_config_sentinel[inverse_indices]

        # If two cubes with cases C16 and C19 share an ambiguous face, both cases are inverted.
        to_invert = (problem_config_adj[..., 0] == 1)
        idx = torch.arange(case_ids.shape[0], dtype=torch.int32, device=self.device)[to_check][within_range][to_invert]
        case_ids.index_put_((idx,), problem_config[to_invert][..., -1])

        return case_ids

    @torch.no_grad()
    def _identify_surf_edges(self, scalar_field, cube_idx, surf_cubes):
        """
        Identifies grid edges that intersect with the underlying surface by checking for opposite signs. As each edge 
        can be shared by multiple cubes, this function also assigns a unique index to each surface-intersecting edge 
        and marks the cube edges with this index.
        """
        occ_n = scalar_field < 0
        all_edges = cube_idx[surf_cubes][:, self.cube_edges].reshape(-1, 2)
        unique_edges, _idx_map, counts = torch.unique(all_edges, dim=0, return_inverse=True, return_counts=True)

        unique_edges = unique_edges.int()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1

        surf_edges_mask = mask_edges[_idx_map]
        counts = counts[_idx_map]

        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.int32, device=cube_idx.device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.int32, device=cube_idx.device)
        # Shaped as [number of cubes x 12 edges per cube]. This is later used to map a cube edge to the unique index
        # for a surface-intersecting edge. Non-surface-intersecting edges are marked with -1.
        idx_map = mapping[_idx_map]
        surf_edges = unique_edges[mask_edges]
        return surf_edges, idx_map, counts, surf_edges_mask

    @torch.no_grad()
    def _identify_surf_cubes(self, scalar_field, cube_idx):
        """
        Identifies grid cubes that intersect with the underlying surface by checking if the signs at 
        all corners are not identical.
        """
        occ_n = scalar_field < 0
        occ_fx8 = occ_n[cube_idx.reshape(-1)].reshape(-1, 8)
        _occ_sum = torch.sum(occ_fx8, -1)
        surf_cubes = (_occ_sum > 0) & (_occ_sum < 8)
        return surf_cubes, occ_fx8

    def _linear_interp(self, edges_weight, edges_x):
        """
        Computes the location of zero-crossings on 'edges_x' using linear interpolation with 'edges_weight'.
        """
        edge_dim = edges_weight.dim() - 2
        assert edges_weight.shape[edge_dim] == 2
        edges_weight = torch.cat([torch.index_select(input=edges_weight, index=torch.tensor(1, device=self.device), dim=edge_dim), -
                                 torch.index_select(input=edges_weight, index=torch.tensor(0, device=self.device), dim=edge_dim)]
                                 , edge_dim)
        denominator = edges_weight.sum(edge_dim)
        ue = (edges_x * edges_weight).sum(edge_dim) / denominator
        return ue

    def _solve_vd_QEF(self, p_bxnx3, norm_bxnx3, c_bx3, qef_reg_scale):
        p_bxnx3 = p_bxnx3.reshape(-1, 7, 3)
        norm_bxnx3 = norm_bxnx3.reshape(-1, 7, 3)
        c_bx3 = c_bx3.reshape(-1, 3)
        A = norm_bxnx3
        B = ((p_bxnx3) * norm_bxnx3).sum(-1, keepdims=True)

        A_reg = (torch.eye(3, device=p_bxnx3.device) * qef_reg_scale).unsqueeze(0).repeat(p_bxnx3.shape[0], 1, 1)
        B_reg = (qef_reg_scale * c_bx3).unsqueeze(-1)
        A = torch.cat([A, A_reg], 1)
        B = torch.cat([B, B_reg], 1)
        dual_verts = torch.linalg.lstsq(A, B).solution.squeeze(-1)
        return dual_verts

    def _compute_vd(self, voxelgrid_vertices, surf_cubes_fx8, surf_edges, scalar_field,
                    case_ids, beta, alpha, gamma_f, idx_map, qef_reg_scale, voxelgrid_colors):
        """
        Computes the location of dual vertices as described in Section 4.2
        """
        alpha_nx12x2 = torch.index_select(input=alpha, index=self.cube_edges, dim=1).reshape(-1, 12, 2)
        surf_edges_x = torch.index_select(input=voxelgrid_vertices, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, 3)
        surf_edges_s = torch.index_select(input=scalar_field, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, 1)
        zero_crossing = self._linear_interp(surf_edges_s, surf_edges_x)
        
        if voxelgrid_colors is not None:
            C = voxelgrid_colors.shape[-1]
            surf_edges_c = torch.index_select(input=voxelgrid_colors, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, C)

        idx_map = idx_map.reshape(-1, 12)
        num_vd = torch.index_select(input=self.num_vd_table, index=case_ids, dim=0)
        edge_group, edge_group_to_vd, edge_group_to_cube, vd_num_edges, vd_gamma = [], [], [], [], []
        
        # if color is not None:
        #     vd_color = []

        total_num_vd = 0
        vd_idx_map = torch.zeros((case_ids.shape[0], 12), dtype=torch.long, device=self.device, requires_grad=False)

        for num in torch.unique(num_vd):
            cur_cubes = (num_vd == num)  # consider cubes with the same numbers of vd emitted (for batching)
            curr_num_vd = cur_cubes.sum() * num
            curr_edge_group = self.dmc_table[case_ids[cur_cubes], :num].reshape(-1, num * 7)
            curr_edge_group_to_vd = torch.arange(
                curr_num_vd, device=self.device).unsqueeze(-1).repeat(1, 7) + total_num_vd
            total_num_vd += curr_num_vd
            curr_edge_group_to_cube = torch.arange(idx_map.shape[0], device=self.device)[
                cur_cubes].unsqueeze(-1).repeat(1, num * 7).reshape_as(curr_edge_group)

            curr_mask = (curr_edge_group != -1)
            edge_group.append(torch.masked_select(curr_edge_group, curr_mask))
            edge_group_to_vd.append(torch.masked_select(curr_edge_group_to_vd.reshape_as(curr_edge_group), curr_mask))
            edge_group_to_cube.append(torch.masked_select(curr_edge_group_to_cube, curr_mask))
            vd_num_edges.append(curr_mask.reshape(-1, 7).sum(-1, keepdims=True))
            vd_gamma.append(torch.masked_select(gamma_f, cur_cubes).unsqueeze(-1).repeat(1, num).reshape(-1))
            # if color is not None:
            #     vd_color.append(color[cur_cubes].unsqueeze(1).repeat(1, num, 1).reshape(-1, 3))
            
        edge_group = torch.cat(edge_group)
        edge_group_to_vd = torch.cat(edge_group_to_vd)
        edge_group_to_cube = torch.cat(edge_group_to_cube)
        vd_num_edges = torch.cat(vd_num_edges)
        vd_gamma = torch.cat(vd_gamma)
        # if color is not None:
        #     vd_color = torch.cat(vd_color)
        # else:
        #     vd_color = None

        vd = torch.zeros((total_num_vd, 3), device=self.device, dtype=beta.dtype)
        beta_sum = torch.zeros((total_num_vd, 1), device=self.device, dtype=beta.dtype)

        idx_group = torch.gather(input=idx_map.reshape(-1), dim=0, index=edge_group_to_cube * 12 + edge_group)

        x_group = torch.index_select(input=surf_edges_x, index=idx_group.reshape(-1), dim=0).reshape(-1, 2, 3)
        s_group = torch.index_select(input=surf_edges_s, index=idx_group.reshape(-1), dim=0).reshape(-1, 2, 1)
        

        zero_crossing_group = torch.index_select(
            input=zero_crossing, index=idx_group.reshape(-1), dim=0).reshape(-1, 3)

        alpha_group = torch.index_select(input=alpha_nx12x2.reshape(-1, 2), dim=0,
                                            index=edge_group_to_cube * 12 + edge_group).reshape(-1, 2, 1)
        ue_group = self._linear_interp(s_group * alpha_group, x_group)

        beta_group = torch.gather(input=beta.reshape(-1), dim=0,
                                    index=edge_group_to_cube * 12 + edge_group).reshape(-1, 1)
        beta_sum = beta_sum.index_add_(0, index=edge_group_to_vd, source=beta_group)
        vd = vd.index_add_(0, index=edge_group_to_vd, source=ue_group * beta_group) / beta_sum
        
        '''
        interpolate colors use the same method as dual vertices
        '''
        if voxelgrid_colors is not None:
            vd_color = torch.zeros((total_num_vd, C), device=self.device, dtype=beta.dtype)
            c_group = torch.index_select(input=surf_edges_c, index=idx_group.reshape(-1), dim=0).reshape(-1, 2, C)
            uc_group = self._linear_interp(s_group * alpha_group, c_group)
            vd_color = vd_color.index_add_(0, index=edge_group_to_vd, source=uc_group * beta_group) / beta_sum
        else:
            vd_color = None
        
        L_dev = self._compute_reg_loss(vd, zero_crossing_group, edge_group_to_vd, vd_num_edges)

        v_idx = torch.arange(vd.shape[0], device=self.device)  # + total_num_vd

        vd_idx_map = (vd_idx_map.reshape(-1)).scatter(dim=0, index=edge_group_to_cube *
                                                      12 + edge_group, src=v_idx[edge_group_to_vd])

        return vd, L_dev, vd_gamma, vd_idx_map, vd_color

    def _triangulate(self, scalar_field, surf_edges, vd, vd_gamma, edge_counts, idx_map, vd_idx_map, surf_edges_mask, vd_color):
        """
        Connects four neighboring dual vertices to form a quadrilateral. The quadrilaterals are then split into
        triangles based on the gamma parameter, as described in Section 4.3.
        """
        with torch.no_grad():
            group_mask = (edge_counts == 4) & surf_edges_mask  # surface edges shared by 4 cubes.
            group = idx_map.reshape(-1)[group_mask]
            vd_idx = vd_idx_map[group_mask]
            edge_indices, indices = torch.sort(group, stable=True)
            indices = indices.to(torch.int32)
            quad_vd_idx = vd_idx[indices].reshape(-1, 4).to(torch.int32)

            # Ensure all face directions point towards the positive SDF to maintain consistent winding.
            s_edges = scalar_field[surf_edges[edge_indices.reshape(-1, 4)[:, 0]].reshape(-1)].reshape(-1, 2)
            flip_mask = s_edges[:, 0] > 0
            quad_vd_idx = torch.cat((quad_vd_idx[flip_mask][:, [0, 1, 3, 2]],
                                     quad_vd_idx[~flip_mask][:, [2, 3, 1, 0]]))

        quad_gamma = torch.index_select(input=vd_gamma, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4)
        gamma_02 = quad_gamma[:, 0] * quad_gamma[:, 2]
        gamma_13 = quad_gamma[:, 1] * quad_gamma[:, 3]
        mask = (gamma_02 > gamma_13)
        faces = torch.zeros((quad_gamma.shape[0], 6), dtype=torch.int32, device=quad_vd_idx.device)
        faces[mask] = quad_vd_idx[mask][:, self.quad_split_1]
        faces[~mask] = quad_vd_idx[~mask][:, self.quad_split_2]
        faces = faces.reshape(-1, 3)
        return vd, faces, s_edges, edge_indices, vd_color

# =============================================================================
# Cube utilities (from representations/mesh/utils_cube.py)
# =============================================================================


cube_corners = torch.tensor(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=torch.int,
)
cube_neighbor = torch.tensor(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
)
cube_edges = torch.tensor(
    [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4],
    dtype=torch.long,
    requires_grad=False,
)



def construct_voxel_grid(coords):
    verts = (cube_corners.unsqueeze(0).to(coords) + coords.unsqueeze(1)).reshape(-1, 3)
    verts_unique, inverse_indices = torch.unique(verts, dim=0, return_inverse=True)
    cubes = inverse_indices.reshape(-1, 8)
    return verts_unique, cubes


def cubes_to_verts(num_verts, cubes, value, reduce="mean"):
    """
    Args:
        cubes [Vx8] verts index for each cube
        value [Vx8xM] value to be scattered
    Operation:
        reduced[cubes[i][j]][k] += value[i][k]
    """
    M = value.shape[2]  # number of channels
    reduced = torch.zeros(num_verts, M, device=cubes.device, dtype=value.dtype)
    return torch.scatter_reduce(
        reduced,
        0,
        cubes.unsqueeze(-1).expand(-1, -1, M).flatten(0, 1),
        value.flatten(0, 1),
        reduce=reduce,
        include_self=False,
    )


def sparse_cube2verts(coords, feats):
    new_coords, cubes = construct_voxel_grid(coords)
    new_feats = cubes_to_verts(new_coords.shape[0], cubes, feats)
    return new_coords, new_feats



def get_sparse_attrs(coords: torch.Tensor, feats: torch.Tensor, res: int, sdf_init=True):
    """Get sparse attrs — returns only occupied coordinates, never materializes a dense [res³, F] grid."""
    verts = coords
    verts, masks = torch.unique(verts, dim=0, return_inverse=True)
    feats_sparse = torch.zeros((len(verts), feats.shape[-1]), device=feats.device, dtype=feats.dtype)
    feats_sparse[masks] = feats
    return feats_sparse, verts


_DILATE_OFFSETS = torch.tensor(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=torch.int,
)


def dilate_sparse_cubes(coords, feats, res):
    """Add 6-connected neighbor cubes with default (zero) features."""
    offsets = _DILATE_OFFSETS.to(device=coords.device, dtype=coords.dtype)
    neighbors = (coords.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)
    valid = (neighbors >= 0).all(-1) & (neighbors < res).all(-1)
    neighbors = neighbors[valid]
    all_coords = torch.cat([coords, neighbors])
    unique_coords, inverse = torch.unique(all_coords, dim=0, return_inverse=True)
    dilated_feats = torch.zeros(len(unique_coords), feats.shape[-1],
                                device=feats.device, dtype=feats.dtype)
    orig_indices = inverse[:len(coords)]
    dilated_feats[orig_indices] = feats
    return unique_coords, dilated_feats


def dilate_sparse_verts(v_pos, v_attrs, dilated_cube_coords, res_v):
    """Add corner vertices for dilated cubes. New verts get SDF=1.0, rest=0."""
    corners = (cube_corners.unsqueeze(0).to(dilated_cube_coords) + dilated_cube_coords.unsqueeze(1)).reshape(-1, 3)
    valid = (corners >= 0).all(-1) & (corners < res_v).all(-1)
    corners = corners[valid]
    all_verts = torch.cat([v_pos, corners])
    unique_verts, inverse = torch.unique(all_verts, dim=0, return_inverse=True)
    dilated_attrs = torch.zeros(len(unique_verts), v_attrs.shape[-1],
                                device=v_attrs.device, dtype=v_attrs.dtype)
    dilated_attrs[..., 0] = 1.0  # SDF=1.0 (outside) default for new verts
    orig_indices = inverse[:len(v_pos)]
    dilated_attrs[orig_indices] = v_attrs  # overwrite with actual data
    return unique_verts, dilated_attrs


def get_defomed_verts(v_pos: torch.Tensor, deform: torch.Tensor, res):
    return (v_pos / res - 0.5 + (1 - 1e-8) / (res * 2) * torch.tanh(deform)).to(
        deform.dtype
    )


# =============================================================================
# SparseFeatures2Mesh / MeshExtractResult (from representations/mesh/cube2mesh.py)
# =============================================================================

class MeshExtractResult:
    def __init__(self, vertices, faces, vertex_attrs=None, res=64):
        self.vertices = vertices.float()
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs.float() if vertex_attrs is not None else None
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = vertices.shape[0] != 0 and faces.shape[0] != 0

    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        # print(face_normals.min(), face_normals.max(), face_normals.shape)
        return face_normals[:, None, :].repeat(1, 3, 1)

    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals


class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=True):
        """
        Generate a mesh from sparse feature structures using SparseFlex (FlexiCubes).
        """
        super().__init__()
        self.device = device
        self.res = res
        self.mesh_extractor = FlexiCubes(device=device)
        self.sdf_bias = -1.0 / res
        self.use_color = use_color
        self._calc_layout()

    def _calc_layout(self):
        LAYOUTS = {
            "sdf": {"shape": (8, 1), "size": 8},
            "deform": {"shape": (8, 3), "size": 8 * 3},
            "weights": {"shape": (21,), "size": 21},
        }
        if self.use_color:
            """
            6 channel color including normal map
            """
            LAYOUTS["color"] = {
                "shape": (
                    8,
                    6,
                ),
                "size": 8 * 6,
            }
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v["range"] = (start, start + v["size"])
            start += v["size"]
        self.feats_channels = start

    def get_layout(self, feats: torch.Tensor, name: str):
        if name not in self.layouts:
            return None
        return feats[
            :, self.layouts[name]["range"][0] : self.layouts[name]["range"][1]
        ].reshape(-1, *self.layouts[name]["shape"])

    def __call__(self, cubefeats: SparseTensor):
        """
        Generates a mesh from sparse voxel structures using SparseFlex.
        """
        import logging
        _log = logging.getLogger("sam3dobjects")
        def _vm():
            return torch.cuda.memory_allocated() / 1024**2

        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        device = coords.device
        _log.warning("[VRAM:mesh] start: %.0f MiB | cubefeats %s", _vm(), cubefeats.feats.shape)

        sdf, deform, color, weights = [
            self.get_layout(feats, name)
            for name in ["sdf", "deform", "color", "weights"]
        ]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs = sparse_cube2verts(
            coords, torch.cat(v_attrs, dim=-1)
        )
        _log.warning("[VRAM:mesh] after sparse_cube2verts: %.0f MiB | v_pos %s", _vm(), v_pos.shape)

        res_v = self.res + 1

        v_attrs_d, v_pos_dilate = get_sparse_attrs(v_pos, v_attrs, res=res_v, sdf_init=True)
        weights_d, coords_dilate = get_sparse_attrs(coords, weights, res=self.res, sdf_init=False)

        # Free input tensor + all extracted views — dilated copies are independent
        cubefeats.data = None  # Break caller's reference to feats (~654 MiB)
        del cubefeats, feats, coords, sdf, deform, color, weights, v_pos, v_attrs
        comfy.model_management.soft_empty_cache()
        _log.warning("[VRAM:mesh] after get_sparse_attrs + del: %.0f MiB | verts %d cubes %d", _vm(), len(v_pos_dilate), len(coords_dilate))

        # Dilate cube set by 1 voxel to add boundary cubes
        coords_dilate, weights_d = dilate_sparse_cubes(coords_dilate, weights_d, self.res)
        v_pos_dilate, v_attrs_d = dilate_sparse_verts(v_pos_dilate, v_attrs_d, coords_dilate, res_v)
        _log.warning("[VRAM:mesh] after dilation: %.0f MiB | verts %d cubes %d", _vm(), len(v_pos_dilate), len(coords_dilate))

        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None

        x_nx3 = get_defomed_verts(v_pos_dilate, deform_d, self.res)
        del deform_d
        comfy.model_management.soft_empty_cache()

        # Append sentinel vertex/sdf for out-of-bounds searchsorted lookups
        x_nx3 = torch.cat((x_nx3, torch.ones((1, 3), dtype=x_nx3.dtype, device=device) * 0.5))
        sdf_d = torch.cat((sdf_d, torch.ones((1,), dtype=sdf_d.dtype, device=device)))
        if colors_d is not None:
            colors_d = torch.cat((colors_d, torch.zeros((1, colors_d.shape[-1]), dtype=colors_d.dtype, device=device)))
        del v_attrs_d
        comfy.model_management.soft_empty_cache()

        # Build sparse cube index via searchsorted
        mask_reg_c_sparse = (v_pos_dilate[..., 0] * res_v + v_pos_dilate[..., 1]) * res_v + v_pos_dilate[..., 2]
        del v_pos_dilate
        comfy.model_management.soft_empty_cache()
        reg_c_sparse = (coords_dilate[..., 0] * res_v + coords_dilate[..., 1]) * res_v + coords_dilate[..., 2]
        cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]
        reg_c_value = (reg_c_sparse.unsqueeze(1) + cube_corners_bias.unsqueeze(0).to(device)).reshape(-1)
        del reg_c_sparse, cube_corners_bias
        reg_c = torch.searchsorted(mask_reg_c_sparse, reg_c_value)
        reg_c = reg_c.clamp(max=len(mask_reg_c_sparse) - 1)
        reg_c[mask_reg_c_sparse[reg_c] != reg_c_value] = len(mask_reg_c_sparse)
        del reg_c_value, mask_reg_c_sparse
        comfy.model_management.soft_empty_cache()
        reg_c = reg_c.reshape(-1, 8)
        _log.warning("[VRAM:mesh] after searchsorted: %.0f MiB | reg_c %s", _vm(), reg_c.shape)

        vertices, faces, _, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            cube_index_map=coords_dilate,
        )

        _log.warning("[VRAM:mesh] after FlexiCubes: %.0f MiB | verts %s faces %s", _vm(), vertices.shape, faces.shape)

        mesh = MeshExtractResult(
            vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res
        )
        return mesh


# =============================================================================
# DfsOctree (from representations/octree/octree_dfs.py)
# =============================================================================

DEFAULT_TRIVEC_CONFIG = {
    "dim": 8,
    "rank": 8,
}

DEFAULT_VOXEL_CONFIG = {
    "solid": False,
}

DEFAULT_DECOPOLY_CONFIG = {
    "degree": 8,
    "rank": 16,
}


class DfsOctree:
    """
    Sparse Voxel Octree (SVO) implementation for PyTorch.
    Using Depth-First Search (DFS) order to store the octree.
    DFS order suits rendering and ray tracing.

    The structure and data are separatedly stored.
    Structure is stored as a continuous array, each element is a 3*32 bits descriptor.
    |-----------------------------------------|
    |      0:3 bits      |      4:31 bits     |
    |      leaf num      |       unused       |
    |-----------------------------------------|
    |               0:31  bits                |
    |                child ptr                |
    |-----------------------------------------|
    |               0:31  bits                |
    |                data ptr                 |
    |-----------------------------------------|
    Each element represents a non-leaf node in the octree.
    The valid mask is used to indicate whether the children are valid.
    The leaf mask is used to indicate whether the children are leaf nodes.
    The child ptr is used to point to the first non-leaf child. Non-leaf children descriptors are stored continuously from the child ptr.
    The data ptr is used to point to the data of leaf children. Leaf children data are stored continuously from the data ptr.

    There are also auxiliary arrays to store the additional structural information to facilitate parallel processing.
      - Position: the position of the octree nodes.
      - Depth: the depth of the octree nodes.

    Args:
        depth (int): the depth of the octree.
    """

    def __init__(
        self,
        depth,
        aabb=[0, 0, 0, 1, 1, 1],
        sh_degree=2,
        primitive="voxel",
        primitive_config={},
        device="cuda",
    ):
        self.max_depth = depth
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.device = device
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.primitive = primitive
        self.primitive_config = primitive_config

        self.structure = torch.tensor(
            [[8, 1, 0]], dtype=torch.int32, device=self.device
        )
        self.position = torch.zeros((8, 3), dtype=torch.float32, device=self.device)
        self.depth = torch.zeros((8, 1), dtype=torch.uint8, device=self.device)
        self.position[:, 0] = torch.tensor(
            [0.25, 0.75, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75], device=self.device
        )
        self.position[:, 1] = torch.tensor(
            [0.25, 0.25, 0.75, 0.75, 0.25, 0.25, 0.75, 0.75], device=self.device
        )
        self.position[:, 2] = torch.tensor(
            [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75], device=self.device
        )
        self.depth[:, 0] = 1

        self.data = ["position", "depth"]
        self.param_names = []

        if primitive == "voxel":
            self.features_dc = torch.zeros(
                (8, 1, 3), dtype=torch.float32, device=self.device
            )
            self.features_ac = torch.zeros(
                (8, (sh_degree + 1) ** 2 - 1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.data += ["features_dc", "features_ac"]
            self.param_names += ["features_dc", "features_ac"]
            if not primitive_config.get("solid", False):
                self.density = torch.zeros(
                    (8, 1), dtype=torch.float32, device=self.device
                )
                self.data.append("density")
                self.param_names.append("density")
        elif primitive == "gaussian":
            self.features_dc = torch.zeros(
                (8, 1, 3), dtype=torch.float32, device=self.device
            )
            self.features_ac = torch.zeros(
                (8, (sh_degree + 1) ** 2 - 1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.opacity = torch.zeros((8, 1), dtype=torch.float32, device=self.device)
            self.data += ["features_dc", "features_ac", "opacity"]
            self.param_names += ["features_dc", "features_ac", "opacity"]
        elif primitive == "trivec":
            self.trivec = torch.zeros(
                (8, primitive_config["rank"], 3, primitive_config["dim"]),
                dtype=torch.float32,
                device=self.device,
            )
            self.density = torch.zeros(
                (8, primitive_config["rank"]), dtype=torch.float32, device=self.device
            )
            self.features_dc = torch.zeros(
                (8, primitive_config["rank"], 1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.features_ac = torch.zeros(
                (8, primitive_config["rank"], (sh_degree + 1) ** 2 - 1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.density_shift = 0
            self.data += ["trivec", "density", "features_dc", "features_ac"]
            self.param_names += ["trivec", "density", "features_dc", "features_ac"]
        elif primitive == "decoupoly":
            self.decoupoly_V = torch.zeros(
                (8, primitive_config["rank"], 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.decoupoly_g = torch.zeros(
                (8, primitive_config["rank"], primitive_config["degree"]),
                dtype=torch.float32,
                device=self.device,
            )
            self.density = torch.zeros(
                (8, primitive_config["rank"]), dtype=torch.float32, device=self.device
            )
            self.features_dc = torch.zeros(
                (8, primitive_config["rank"], 1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.features_ac = torch.zeros(
                (8, primitive_config["rank"], (sh_degree + 1) ** 2 - 1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self.density_shift = 0
            self.data += [
                "decoupoly_V",
                "decoupoly_g",
                "density",
                "features_dc",
                "features_ac",
            ]
            self.param_names += [
                "decoupoly_V",
                "decoupoly_g",
                "density",
                "features_dc",
                "features_ac",
            ]

        self.setup_functions()

    def setup_functions(self):
        self.density_activation = (
            (lambda x: torch.exp(x - 2))
            if self.primitive != "trivec"
            else (lambda x: x)
        )
        self.opacity_activation = lambda x: torch.sigmoid(x - 6)
        self.inverse_opacity_activation = lambda x: torch.log(x / (1 - x)) + 6
        self.color_activation = lambda x: torch.sigmoid(x)

    @property
    def num_non_leaf_nodes(self):
        return self.structure.shape[0]

    @property
    def num_leaf_nodes(self):
        return self.depth.shape[0]

    @property
    def cur_depth(self):
        return self.depth.max().item()

    @property
    def occupancy(self):
        return self.num_leaf_nodes / 8**self.cur_depth

    @property
    def get_xyz(self):
        return self.position

    @property
    def get_depth(self):
        return self.depth

    @property
    def get_density(self):
        if self.primitive == "voxel" and self.voxel_config["solid"]:
            return torch.full(
                (self.position.shape[0], 1),
                1000,
                dtype=torch.float32,
                device=self.device,
            )
        return self.density_activation(self.density)

    @property
    def get_opacity(self):
        return self.opacity_activation(self.density)

    @property
    def get_trivec(self):
        return self.trivec

    @property
    def get_decoupoly(self):
        return F.normalize(self.decoupoly_V, dim=-1), self.decoupoly_g

    @property
    def get_color(self):
        return self.color_activation(self.colors)

    @property
    def get_features(self):
        if self.sh_degree == 0:
            return self.features_dc
        return torch.cat([self.features_dc, self.features_ac], dim=-2)

    def state_dict(self):
        ret = {
            "structure": self.structure,
            "position": self.position,
            "depth": self.depth,
            "sh_degree": self.sh_degree,
            "active_sh_degree": self.active_sh_degree,
            "trivec_config": self.trivec_config,
            "voxel_config": self.voxel_config,
            "primitive": self.primitive,
        }
        if hasattr(self, "density_shift"):
            ret["density_shift"] = self.density_shift
        for data in set(self.data + self.param_names):
            if not isinstance(getattr(self, data), nn.Module):
                ret[data] = getattr(self, data)
            else:
                ret[data] = getattr(self, data).state_dict()
        return ret

    def load_state_dict(self, state_dict):
        keys = list(
            set(
                self.data
                + self.param_names
                + list(state_dict.keys())
                + ["structure", "position", "depth"]
            )
        )
        for key in keys:
            if key not in state_dict:
                log.warning("key %s not found in the state_dict.", key)
                continue
            try:
                if not isinstance(getattr(self, key), nn.Module):
                    setattr(self, key, state_dict[key])
                else:
                    getattr(self, key).load_state_dict(state_dict[key])
            except Exception as e:
                log.error("%s", e)
                raise ValueError(f"Error loading key {key}.")

    def gather_from_leaf_children(self, data):
        """
        Gather the data from the leaf children.

        Args:
            data (torch.Tensor): the data to gather. The first dimension should be the number of leaf nodes.
        """
        leaf_cnt = self.structure[:, 0]
        leaf_cnt_masks = [leaf_cnt == i for i in range(1, 9)]
        ret = torch.zeros(
            (self.num_non_leaf_nodes,), dtype=data.dtype, device=self.device
        )
        for i in range(8):
            if leaf_cnt_masks[i].sum() == 0:
                continue
            start = self.structure[leaf_cnt_masks[i], 2]
            for j in range(i + 1):
                ret[leaf_cnt_masks[i]] += data[start + j]
        return ret

    def gather_from_non_leaf_children(self, data):
        """
        Gather the data from the non-leaf children.

        Args:
            data (torch.Tensor): the data to gather. The first dimension should be the number of leaf nodes.
        """
        non_leaf_cnt = 8 - self.structure[:, 0]
        non_leaf_cnt_masks = [non_leaf_cnt == i for i in range(1, 9)]
        ret = torch.zeros_like(data, device=self.device)
        for i in range(8):
            if non_leaf_cnt_masks[i].sum() == 0:
                continue
            start = self.structure[non_leaf_cnt_masks[i], 1]
            for j in range(i + 1):
                ret[non_leaf_cnt_masks[i]] += data[start + j]
        return ret

    def structure_control(self, mask):
        """
        Control the structure of the octree.

        Args:
            mask (torch.Tensor): the mask to control the structure. 1 for subdivide, -1 for merge, 0 for keep.
        """
        # Dont subdivide when the depth is the maximum.
        mask[self.depth.squeeze() == self.max_depth] = torch.clamp_max(
            mask[self.depth.squeeze() == self.max_depth], 0
        )
        # Dont merge when the depth is the minimum.
        mask[self.depth.squeeze() == 1] = torch.clamp_min(
            mask[self.depth.squeeze() == 1], 0
        )

        # Gather control mask
        structre_ctrl = self.gather_from_leaf_children(mask)
        structre_ctrl[structre_ctrl == -8] = -1

        new_leaf_num = self.structure[:, 0].clone()
        # Modify the leaf num.
        structre_valid = structre_ctrl >= 0
        new_leaf_num[structre_valid] -= structre_ctrl[
            structre_valid
        ]  # Add the new nodes.
        structre_delete = structre_ctrl < 0
        merged_nodes = self.gather_from_non_leaf_children(structre_delete.int())
        new_leaf_num += merged_nodes  # Delete the merged nodes.

        # Update the structure array to allocate new nodes.
        mem_offset = torch.zeros(
            (self.num_non_leaf_nodes + 1,), dtype=torch.int32, device=self.device
        )
        mem_offset.index_add_(
            0, self.structure[structre_valid, 1], structre_ctrl[structre_valid]
        )  # Add the new nodes.
        mem_offset[:-1] -= structre_delete.int()  # Delete the merged nodes.
        new_structre_idx = torch.arange(
            0, self.num_non_leaf_nodes + 1, dtype=torch.int32, device=self.device
        ) + mem_offset.cumsum(0)
        new_structure_length = new_structre_idx[-1].item()
        new_structre_idx = new_structre_idx[:-1]
        new_structure = torch.empty(
            (new_structure_length, 3), dtype=torch.int32, device=self.device
        )
        new_structure[new_structre_idx[structre_valid], 0] = new_leaf_num[
            structre_valid
        ]

        # Initialize the new nodes.
        new_node_mask = torch.ones(
            (new_structure_length,), dtype=torch.bool, device=self.device
        )
        new_node_mask[new_structre_idx[structre_valid]] = False
        new_structure[new_node_mask, 0] = 8  # Initialize to all leaf nodes.
        new_node_num = new_node_mask.sum().item()

        # Rebuild child ptr.
        non_leaf_cnt = 8 - new_structure[:, 0]
        new_child_ptr = torch.cat(
            [
                torch.zeros((1,), dtype=torch.int32, device=self.device),
                non_leaf_cnt.cumsum(0)[:-1],
            ]
        )
        new_structure[:, 1] = new_child_ptr + 1

        # Rebuild data ptr with old data.
        leaf_cnt = torch.zeros(
            (new_structure_length,), dtype=torch.int32, device=self.device
        )
        leaf_cnt.index_add_(0, new_structre_idx, self.structure[:, 0])
        old_data_ptr = torch.cat(
            [
                torch.zeros((1,), dtype=torch.int32, device=self.device),
                leaf_cnt.cumsum(0)[:-1],
            ]
        )

        # Update the data array
        subdivide_mask = mask == 1
        merge_mask = mask == -1
        data_valid = ~(subdivide_mask | merge_mask)
        mem_offset = torch.zeros(
            (self.num_leaf_nodes + 1,), dtype=torch.int32, device=self.device
        )
        mem_offset.index_add_(
            0,
            old_data_ptr[new_node_mask],
            torch.full((new_node_num,), 8, dtype=torch.int32, device=self.device),
        )  # Add data array for new nodes
        mem_offset[
            :-1
        ] -= subdivide_mask.int()  # Delete data elements for subdivide nodes
        mem_offset[:-1] -= merge_mask.int()  # Delete data elements for merge nodes
        mem_offset.index_add_(
            0, self.structure[structre_valid, 2], merged_nodes[structre_valid]
        )  # Add data elements for merge nodes
        new_data_idx = torch.arange(
            0, self.num_leaf_nodes + 1, dtype=torch.int32, device=self.device
        ) + mem_offset.cumsum(0)
        new_data_length = new_data_idx[-1].item()
        new_data_idx = new_data_idx[:-1]
        new_data = {
            data: torch.empty(
                (new_data_length,) + getattr(self, data).shape[1:],
                dtype=getattr(self, data).dtype,
                device=self.device,
            )
            for data in self.data
        }
        for data in self.data:
            new_data[data][new_data_idx[data_valid]] = getattr(self, data)[data_valid]

        # Rebuild data ptr
        leaf_cnt = new_structure[:, 0]
        new_data_ptr = torch.cat(
            [
                torch.zeros((1,), dtype=torch.int32, device=self.device),
                leaf_cnt.cumsum(0)[:-1],
            ]
        )
        new_structure[:, 2] = new_data_ptr

        # Initialize the new data array
        ## For subdivide nodes
        if subdivide_mask.sum() > 0:
            subdivide_data_ptr = new_structure[new_node_mask, 2]
            for data in self.data:
                for i in range(8):
                    if data == "position":
                        offset = (
                            torch.tensor(
                                [i // 4, (i // 2) % 2, i % 2],
                                dtype=torch.float32,
                                device=self.device,
                            )
                            - 0.5
                        )
                        scale = 2 ** (-1.0 - self.depth[subdivide_mask])
                        new_data["position"][subdivide_data_ptr + i] = (
                            self.position[subdivide_mask] + offset * scale
                        )
                    elif data == "depth":
                        new_data["depth"][subdivide_data_ptr + i] = (
                            self.depth[subdivide_mask] + 1
                        )
                    elif data == "opacity":
                        new_data["opacity"][subdivide_data_ptr + i] = (
                            self.inverse_opacity_activation(
                                torch.sqrt(
                                    self.opacity_activation(
                                        self.opacity[subdivide_mask]
                                    )
                                )
                            )
                        )
                    elif data == "trivec":
                        offset = (
                            torch.tensor(
                                [i // 4, (i // 2) % 2, i % 2],
                                dtype=torch.float32,
                                device=self.device,
                            )
                            * 0.5
                        )
                        coord = (
                            torch.linspace(
                                0,
                                0.5,
                                self.trivec.shape[-1],
                                dtype=torch.float32,
                                device=self.device,
                            )[None]
                            + offset[:, None]
                        ).reshape(1, 3, self.trivec.shape[-1], 1)
                        axis = (
                            torch.linspace(
                                0, 1, 3, dtype=torch.float32, device=self.device
                            )
                            .reshape(1, 3, 1, 1)
                            .repeat(1, 1, self.trivec.shape[-1], 1)
                        )
                        coord = (
                            torch.stack([coord, axis], dim=3)
                            .reshape(1, 3, self.trivec.shape[-1], 2)
                            .expand(self.trivec[subdivide_mask].shape[0], -1, -1, -1)
                            * 2
                            - 1
                        )
                        new_data["trivec"][subdivide_data_ptr + i] = F.grid_sample(
                            self.trivec[subdivide_mask], coord, align_corners=True
                        )
                    else:
                        new_data[data][subdivide_data_ptr + i] = getattr(self, data)[
                            subdivide_mask
                        ]
        ## For merge nodes
        if merge_mask.sum() > 0:
            merge_data_ptr = torch.empty(
                (merged_nodes.sum().item(),), dtype=torch.int32, device=self.device
            )
            merge_nodes_cumsum = torch.cat(
                [
                    torch.zeros((1,), dtype=torch.int32, device=self.device),
                    merged_nodes.cumsum(0)[:-1],
                ]
            )
            for i in range(8):
                merge_data_ptr[merge_nodes_cumsum[merged_nodes > i] + i] = (
                    new_structure[new_structre_idx[merged_nodes > i], 2] + i
                )
            old_merge_data_ptr = self.structure[structre_delete, 2]
            for data in self.data:
                if data == "position":
                    scale = 2 ** (1.0 - self.depth[old_merge_data_ptr])
                    new_data["position"][merge_data_ptr] = (
                        ((self.position[old_merge_data_ptr] + 0.5) / scale).floor()
                        * scale
                        + 0.5 * scale
                        - 0.5
                    )
                elif data == "depth":
                    new_data["depth"][merge_data_ptr] = (
                        self.depth[old_merge_data_ptr] - 1
                    )
                elif data == "opacity":
                    new_data["opacity"][subdivide_data_ptr + i] = (
                        self.inverse_opacity_activation(
                            self.opacity_activation(self.opacity[subdivide_mask]) ** 2
                        )
                    )
                elif data == "trivec":
                    new_data["trivec"][merge_data_ptr] = self.trivec[old_merge_data_ptr]
                else:
                    new_data[data][merge_data_ptr] = getattr(self, data)[
                        old_merge_data_ptr
                    ]

        # Update the structure and data array
        self.structure = new_structure
        for data in self.data:
            setattr(self, data, new_data[data])

        # Save data array control temp variables
        self.data_rearrange_buffer = {
            "subdivide_mask": subdivide_mask,
            "merge_mask": merge_mask,
            "data_valid": data_valid,
            "new_data_idx": new_data_idx,
            "new_data_length": new_data_length,
            "new_data": new_data,
        }


# =============================================================================
# Strivec (from representations/radiance_field/strivec.py)
# =============================================================================

# Alias for Strivec to reference DfsOctree as its parent
Octree = DfsOctree

class Strivec(Octree):
    def __init__(
        self,
        resolution: int,
        aabb: list,
        sh_degree: int = 0,
        rank: int = 8,
        dim: int = 8,
        device: str = "cuda",
    ):
        assert np.log2(resolution) % 1 == 0, "Resolution must be a power of 2"
        self.resolution = resolution
        depth = int(np.round(np.log2(resolution)))
        super().__init__(
            depth=depth,
            aabb=aabb,
            sh_degree=sh_degree,
            primitive="trivec",
            primitive_config={"rank": rank, "dim": dim},
            device=device,
        )
