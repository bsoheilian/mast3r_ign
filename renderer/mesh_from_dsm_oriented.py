from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras, TexturesUV
from pytorch3d.structures import Meshes
from scipy.ndimage import map_coordinates

if __package__:
	from .mesh_from_dsm import mesh_from_DSM, _read_rgb_image
else:
	from mesh_from_dsm import mesh_from_DSM, _read_rgb_image

if TYPE_CHECKING:
	from dchan.core.utils.lamb93_euc_local import Lamb93_Euc_Local
	from dchan.core.utils.stereopolis_orientation import STOrientation


class OrientedImageTexturedDSMMesh(mesh_from_DSM):
	"""Build a local Euclidean DSM mesh textured from one undistorted image.

	Source and target STOrientations must already use the supplied local frame
	and identity distortion. DSM heights must match the camera altitude datum.
	Xg/Yg/Zg and decode_xyz() contain full local coordinates; vertices are
	centered float32 rendering coordinates. CUDA is required, as in the parent.

	Texture coverage is checked at grid nodes, without source-camera occlusion
	detection. Faces lacking coverage at any vertex are omitted. Texture detail
	is limited by DSM resolution; vertical facades and overhangs are not added.
	"""

	_chunk_size = 65536

	def __init__(
		self,
		dsm_tiff_path: Union[str, Path],
		vrt_path: Union[str, Path],
		euclidien_transform: "Lamb93_Euc_Local",
		nodata_value: Optional[float] = None,
	):
		if euclidien_transform is None:
			raise ValueError("A shared euclidien_transform is required.")
		self.euclidien_transform = euclidien_transform
		self._frame_center = self._get_frame_center(euclidien_transform)
		super().__init__(dsm_tiff_path, vrt_path, nodata_value)
		if self.H < 2 or self.W < 2:
			raise ValueError("DSM dimensions must both be at least 2 to build faces.")
		self.valid_mask &= torch.isfinite(self.Xg) & torch.isfinite(self.Yg) & torch.isfinite(self.Zg)
		self._transform_geometry()
		self.projected_pixels: Optional[np.ndarray] = None
		self.baked_texture: Optional[torch.Tensor] = None
		self.texture_valid_mask: Optional[torch.Tensor] = None

	@staticmethod
	def _get_frame_center(transform) -> np.ndarray:
		center = np.array([
			transform.euc_center_x, transform.euc_center_y, transform.euc_center_z,
		], dtype=np.float64)
		if not np.isfinite(center).all():
			raise ValueError("Local frame center must be finite.")
		return center

	def _validate_orientation(self, orientation: "STOrientation") -> None:
		"""Check available frame metadata; never change or reframe a camera."""
		transform = getattr(orientation, "euclidien_transform", None)
		if transform is not None and not np.allclose(
			self._get_frame_center(transform), self._frame_center, atol=1e-6, rtol=0,
		):
			raise ValueError("Orientation and surface must use the same local Euclidean frame.")
		distortion = orientation.distortion_polynomial
		values = (distortion.r3, distortion.r5, distortion.r7, distortion.pps_c, distortion.pps_l)
		if all(value is not None for value in values) and any(
			value != 0 for value in values[:3]
		):
			raise ValueError("Use an undistorted image with identity distortion.")
		width, height = orientation.intrinsic.image_size
		ppa = orientation.intrinsic.ppa
		if width <= 0 or height <= 0 or not np.isfinite([ppa["f"], ppa["cx"], ppa["cy"]]).all() or ppa["f"] <= 0:
			raise ValueError("Orientation must have positive image dimensions and valid intrinsics.")

	def _transform_geometry(self) -> None:
		coordinates = [self.Xg.reshape(-1), self.Yg.reshape(-1), self.Zg.reshape(-1)]
		valid = self.valid_mask.reshape(-1)
		for start in range(0, valid.numel(), self._chunk_size):
			stop = min(start + self._chunk_size, valid.numel())
			indices = torch.nonzero(valid[start:stop], as_tuple=False).flatten() + start
			if indices.numel() == 0:
				continue
			world = [coordinate[indices].cpu().numpy() for coordinate in coordinates]
			local = np.stack([
				np.asarray(axis, dtype=np.float64).reshape(-1)
				for axis in self.euclidien_transform.lamb93_to_euclidean(*world)
			], axis=1)
			valid[indices] = torch.as_tensor(np.isfinite(local).all(axis=1), device=self.device)
			for axis, coordinate in enumerate(coordinates):
				coordinate[indices] = torch.as_tensor(local[:, axis], device=self.device)
		if not self.valid_mask.any().item():
			raise ValueError("DSM contains no valid local geometry.")
		for coordinate in coordinates:
			coordinate[~valid] = torch.nan

	def project_vertices_to_image(self, source_orientation: "STOrientation") -> np.ndarray:
		"""Return (H, W, 2) source pixel coordinates, with NaNs outside coverage.

		Projection uses full local coordinates, never centered mesh vertices.
		Missing frame metadata leaves same-frame consistency to the caller.
		"""
		self._validate_orientation(source_orientation)
		width, height = source_orientation.intrinsic.image_size
		projected = np.full((self.H * self.W, 2), np.nan, dtype=np.float64)
		coordinates = [self.Xg.reshape(-1), self.Yg.reshape(-1), self.Zg.reshape(-1)]
		valid = self.valid_mask.reshape(-1)
		for start in range(0, valid.numel(), self._chunk_size):
			stop = min(start + self._chunk_size, valid.numel())
			indices = torch.nonzero(valid[start:stop], as_tuple=False).flatten() + start
			if indices.numel() == 0:
				continue
			points = torch.stack([coordinate[indices] for coordinate in coordinates], dim=1).cpu().numpy()
			for index, point in zip(indices.cpu().numpy(), points):
				pixel = source_orientation.LocalToImage(point, check_FOV=True)
				if np.isfinite(pixel).all() and 0 <= pixel[0] <= width - 1 and 0 <= pixel[1] <= height - 1:
					projected[index] = pixel
		return projected.reshape(self.H, self.W, 2)

	def build_textured_mesh(
		self,
		source_image: Union[str, Path, np.ndarray, torch.Tensor],
		source_orientation: "Optional[STOrientation]" = None,
		texture_sampling_mode: str = "bilinear",
	) -> Meshes:
		"""Bake one source image onto the DSM grid and replace the textured mesh.

		RGB inputs must be HWC uint8 or floating values in [0, 1] or [0, 255].
		Image dimensions must match the source calibration. Every retained face
		has three covered vertices. No source visibility/occlusion is computed.
		"""
		if source_orientation is None:
			raise ValueError("source_orientation is required to project the texture.")
		if texture_sampling_mode not in ("nearest", "bilinear"):
			raise ValueError("texture_sampling_mode must be 'nearest' or 'bilinear'.")
		self._validate_orientation(source_orientation)
		if isinstance(source_image, (str, Path)):
			rgb = _read_rgb_image(Path(source_image))
		elif isinstance(source_image, torch.Tensor):
			rgb = source_image.detach().cpu().numpy()
		elif isinstance(source_image, np.ndarray):
			rgb = source_image
		else:
			raise TypeError("source_image must be a path, numpy array, or torch tensor.")
		width, height = source_orientation.intrinsic.image_size
		if rgb.shape != (height, width, 3):
			raise ValueError(f"Source RGB must have shape {(height, width, 3)}, got {rgb.shape}.")
		is_uint8 = rgb.dtype == np.uint8
		if not is_uint8 and not np.issubdtype(rgb.dtype, np.floating):
			raise ValueError("Source RGB must be uint8 or floating point.")
		rgb = rgb.astype(np.float32)
		if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
			raise ValueError("Source RGB must be finite and within [0, 1] or [0, 255].")
		if is_uint8 or rgb.max() > 1:
			rgb /= 255.0

		projected = self.project_vertices_to_image(source_orientation)
		texture_valid = np.isfinite(projected).all(axis=2)
		baked = np.zeros((self.H, self.W, 3), dtype=np.float32)
		sample_coordinates = projected[texture_valid][:, ::-1].T
		for channel in range(3):
			baked[..., channel][texture_valid] = map_coordinates(
				rgb[..., channel], sample_coordinates,
				order=1 if texture_sampling_mode == "bilinear" else 0,
				mode="constant", cval=0.0, prefilter=False,
			)
		texture_valid_tensor = torch.as_tensor(texture_valid, device=self.device)
		baked_tensor = torch.as_tensor(baked, device=self.device)

		rows, cols = torch.meshgrid(
			torch.arange(self.H - 1, device=self.device),
			torch.arange(self.W - 1, device=self.device), indexing="ij",
		)
		top_left = rows * self.W + cols
		top_right = top_left + 1
		bottom_left = top_left + self.W
		bottom_right = bottom_left + 1
		faces = torch.stack([
			torch.stack([top_left, bottom_left, top_right], dim=-1),
			torch.stack([top_right, bottom_left, bottom_right], dim=-1),
		], dim=0).reshape(-1, 3)
		node_valid = (self.valid_mask & texture_valid_tensor).reshape(-1)
		faces = faces[node_valid[faces].all(dim=1)]
		if faces.shape[0] == 0:
			raise ValueError("No DSM faces have source-image coverage at all three vertices.")

		origin = torch.stack([
			torch.median(coordinate[self.valid_mask])
			for coordinate in (self.Xg, self.Yg, self.Zg)
		])
		vertices = torch.stack([self.Xg, self.Yg, self.Zg], dim=-1) - origin
		vertices[~self.valid_mask] = 0.0
		vertices = vertices.reshape(-1, 3).float()
		vertical, horizontal = torch.meshgrid(
			torch.linspace(1.0, 0.0, self.H, device=self.device),
			torch.linspace(0.0, 1.0, self.W, device=self.device), indexing="ij",
		)
		verts_uvs = torch.stack([horizontal, vertical], dim=-1).reshape(-1, 2)
		textures = TexturesUV(
			maps=baked_tensor.unsqueeze(0), verts_uvs=verts_uvs.unsqueeze(0),
			faces_uvs=faces.unsqueeze(0), sampling_mode=texture_sampling_mode,
			align_corners=True,
		)
		mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
		self.projected_pixels = projected
		self.baked_texture = baked_tensor
		self.texture_valid_mask = texture_valid_tensor
		self.render_origin_world = origin
		self.vertices = vertices
		self.faces = faces
		self.mesh = mesh
		return mesh

	def create_perspective_camera_from_rt(
		self,
		R: Union[np.ndarray, torch.Tensor],
		T: Union[np.ndarray, torch.Tensor, Tuple[float, float, float]],
		focal_length_px: Union[float, Tuple[float, float]],
		principal_point_px: Tuple[float, float],
		image_size: Tuple[int, int],
	) -> PerspectiveCameras:
		"""Use image-to-local R and full local camera center T, with size (H, W).

		Supply camera centers as float64 to preserve geospatial precision.
		The inherited render_origin_world name denotes a local-frame origin here.
		"""
		if self.render_origin_world is None:
			raise RuntimeError("Call build_textured_mesh(...) before creating camera.")
		rotation = torch.as_tensor(R, device=self.device, dtype=torch.float32)
		center = torch.as_tensor(T, device=self.device, dtype=torch.float64)
		if rotation.ndim == 2:
			rotation = rotation.unsqueeze(0)
		if center.ndim == 1:
			center = center.unsqueeze(0)
		if rotation.shape != (1, 3, 3) or center.shape != (1, 3):
			raise ValueError("Expected R with shape (3,3) or (1,3,3) and T with shape (3,) or (1,3).")
		if not torch.isfinite(rotation).all().item() or not torch.isfinite(center).all().item():
			raise ValueError("Camera rotation and center must be finite.")
		flip_xy = torch.diag(torch.tensor(
			[-1.0, -1.0, 1.0], device=self.device, dtype=torch.float32,
		)).unsqueeze(0)
		converted_rotation = torch.bmm(rotation, flip_xy)
		center_relative = (center - self.render_origin_world.to(dtype=torch.float64).unsqueeze(0)).float()
		translation = -torch.bmm(center_relative.unsqueeze(1), converted_rotation).squeeze(1)
		if isinstance(focal_length_px, tuple):
			focal = tuple(float(value) for value in focal_length_px)
		else:
			focal = (float(focal_length_px), float(focal_length_px))
		principal = tuple(float(value) for value in principal_point_px)
		height, width = int(image_size[0]), int(image_size[1])
		if height <= 0 or width <= 0 or not np.isfinite(focal + principal).all() or min(focal) <= 0:
			raise ValueError("Camera image size and focal lengths must be positive; intrinsics must be finite.")
		return PerspectiveCameras(
			device=self.device, R=converted_rotation, T=translation,
			focal_length=(focal,), principal_point=(principal,),
			image_size=((height, width),), in_ndc=False,
		)

	def create_perspective_camera_from_orientation(
		self, target_orientation: "STOrientation",
	) -> PerspectiveCameras:
		"""Read the target pose and calibration without changing the baked texture."""
		self._validate_orientation(target_orientation)
		width, height = target_orientation.intrinsic.image_size
		ppa = target_orientation.intrinsic.ppa
		return self.create_perspective_camera_from_rt(
			R=target_orientation.extrinsic.get_rot(image2ground=True),
			T=target_orientation.extrinsic.get_center(),
			focal_length_px=ppa["f"], principal_point_px=(ppa["cx"], ppa["cy"]),
			image_size=(height, width),
		)

	def render_from_orientation(
		self,
		target_orientation: "STOrientation",
		faces_per_pixel: int = 1,
		blur_radius: float = 0.0,
		cull_backfaces: bool = False,
		bin_size: Optional[int] = None,
		max_faces_per_bin: Optional[int] = None,
		render_mode: str = "texture_only",
		enforce_cuda_outputs: bool = False,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Render RGB and camera-axis depth from either an aerial or ground camera.

		Reuse this mesh for multiple target views. To reverse image roles, rebuild
		with the other source image/orientation, then render the former source view.
		"""
		camera = self.create_perspective_camera_from_orientation(target_orientation)
		width, height = target_orientation.intrinsic.image_size
		return self.render_perspective(
			camera=camera, image_size=(height, width), faces_per_pixel=faces_per_pixel,
			blur_radius=blur_radius, cull_backfaces=cull_backfaces, bin_size=bin_size,
			max_faces_per_bin=max_faces_per_bin, render_mode=render_mode,
			enforce_cuda_outputs=enforce_cuda_outputs,
		)

if __name__ == "__main__":
	from dchan.core.utils.stereopolis_orientation import STOrientation

	if __package__:
		from .mesh_from_dsm import (
			_save_rgb_image, _save_depth_image, _save_depth_preview_image, _show_rendered,
		)
	else:
		from mesh_from_dsm import (
			_save_rgb_image, _save_depth_image, _save_depth_preview_image, _show_rendered,
		)

	root_path = "/mast3r_ign/data/chantier_paris/preview_301/mast3r_sample_data/Paris-140418_0545-301-00002_0000673"
	dsm_path = Path(root_path) / "Paris-140418_0545-301-00002_0000673_dsm_cropped.tif"
	vrt_path = Path(root_path) / "Paris-140418_0545-301-00002_0000673_dsm_cropped.vrt"
	source_ori_path = Path(root_path) / "pva_thumbnails/FAUUPARIS10x00026_24551_19052014_675_3466.xml"
	source_ori = STOrientation(source_ori_path)
	source_image_path = Path(root_path) / "pva_thumbnails/FAUUPARIS10x00026_24551_19052014_675_3466.tif"
	surface = OrientedImageTexturedDSMMesh(dsm_path, vrt_path, source_ori.euclidien_transform)
	surface.build_textured_mesh(source_image_path, source_ori)
	target_ori_path = Path(root_path) / "Paris-140418_0545-301-00002_0000673_upright_euc.xml"
	target_ori = STOrientation(target_ori_path)
	rgb, depth = surface.render_from_orientation(target_ori)

	output_dir = Path(root_path) / "rendered_views"
	rgb_path = output_dir / "rgb.png"
	depth_path = output_dir / "depth.tif"
	depth_preview_path = output_dir / "depth.preview.png"
	_save_rgb_image(rgb, rgb_path)
	_save_depth_image(depth, depth_path)
	_save_depth_preview_image(depth, depth_preview_path)
	print(f"Saved RGB: {rgb_path}")
	print(f"Saved depth: {depth_path}")
	print(f"Saved depth preview: {depth_preview_path}")
	_show_rendered(rgb, depth)

#it runs the script from the command line
#problems : 
# resolution problem , the dsm seems to be badly textured , low resolution and also georef problem

#to run this script from the command line, use the following commands:
#export PYTHONPATH="/dchan:/mast3r_ign${PYTHONPATH:+:$PYTHONPATH}"
#python /mast3r_ign/renderer/mesh_from_dsm_oriented.py

         