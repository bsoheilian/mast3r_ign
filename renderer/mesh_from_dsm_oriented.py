"""Texture a local DSM surface from one oriented image and render another view.

Usage: construct OrientedImageTexturedDSMMesh with the shared Euclidean frame,
call build_textured_mesh(source_image, source_orientation), then
render_from_orientation(target_orientation). Requires CUDA, PyTorch3D and dchan.

Surface inputs are GeoTIFF/VRT paths or elevation arrays/tensors plus a raster
profile. For an already prepared Mast3rTrainingSample, no VRT needs writing:
	surface = OrientedImageTexturedDSMMesh(
		sample.dsm_cropped, sample.upright_euc_orientation.euclidien_transform,
		profile=sample.dsm_cropped_profile,
	)
	surface.build_textured_mesh(aerial_image_path, aerial_orientation_path)
	rgb, depth = surface.render_from_orientation(sample.upright_euc_orientation)
The reverse workflow can texture from sample.upright_image (CHW) and its
upright_euc_orientation. Image and orientation inputs may independently be
objects or paths. Arrays and profiles are never modified by the renderer.

Inside Docker, run the example with:
	export PYTHONPATH="/dchan:/mast3r_ign${PYTHONPATH:+:$PYTHONPATH}"
	python /mast3r_ign/renderer/mesh_from_dsm_oriented.py
"""

from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras, TexturesUV
from pytorch3d.structures import Meshes

if __package__:
	from . import mesh_utils as utils
else:
	import mesh_utils as utils

from dchan.core.utils.lamb93_euc_local import Lamb93_Euc_Local
from dchan.core.utils.stereopolis_orientation import STOrientation


class OrientedImageTexturedDSMMesh:
	"""Lambert93 DSM -> local XYZ -> source UV texture -> target RGB/depth.

	Both STOrientations must use the supplied local frame and identity distortion;
	DSM heights must match their altitude datum. Xg/Yg/Zg are full float64 local
	coordinates; mesh vertices are centered float32. render_origin_world denotes
	the local-frame offset subtracted from both geometry and camera centers.

	Faces need three geometry-valid, source-covered vertices. UVs sample the
	original image, approximating source perspective within each triangle. No
	source occlusion, vertical facade geometry, or overhangs are reconstructed.
	"""

	_chunk_size = 65536

	def __init__(
		self,
		surface: Union[str, Path, np.ndarray, torch.Tensor],
		euclidien_transform: Union[Lamb93_Euc_Local, str, Path],
		*,
		profile: Optional[dict] = None,
		vrt_path: Optional[Union[str, Path]] = None,
		nodata_value: Optional[float] = None,
	):
		"""Load a Lambert93 surface or use (H,W)/(1,H,W) elevations plus profile.

		Memory profiles need an Affine transform; file rasters supply their own
		metadata, optionally overridden by vrt_path. nodata_value overrides
		metadata when not None. The local frame is an object or frame XML path.
		"""
		self.euclidien_transform = self._load_euclidien_transform(euclidien_transform)
		self._frame_center = self._get_frame_center(self.euclidien_transform)
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA is not available. OrientedImageTexturedDSMMesh requires CUDA.")
		self.device = torch.device("cuda")
		project_root = Path(__file__).resolve().parent.parent
		self.dsm_tiff_path = utils.resolve_existing_path(surface, project_root) if isinstance(surface, (str, Path)) else None
		self.vrt_path = utils.resolve_existing_path(vrt_path, project_root) if vrt_path is not None else None
		elevations, self.surface_profile = utils.load_surface(surface, profile=profile, vrt_path=vrt_path)
		self._geotransform = self.surface_profile["transform"].to_gdal()
		self.nodata_value = self.surface_profile.get("nodata") if nodata_value is None else nodata_value
		self.Zg = elevations.to(self.device)
		self.Xg, self.Yg = utils.decode_xy_from_geotransform(self.Zg.shape, self._geotransform, self.device)
		self.H, self.W = self.Zg.shape
		if self.H < 2 or self.W < 2:
			raise ValueError("DSM dimensions must both be at least 2 to build faces.")
		self.valid_mask = utils.build_valid_mask(self.Zg, self.nodata_value)
		self.valid_mask &= torch.isfinite(self.Xg) & torch.isfinite(self.Yg) & torch.isfinite(self.Zg)
		self._transform_geometry()
		self.mesh: Optional[Meshes] = None
		self.vertices: Optional[torch.Tensor] = None
		self.faces: Optional[torch.Tensor] = None
		self.render_origin_world: Optional[torch.Tensor] = None
		self.projected_pixels: Optional[np.ndarray] = None
		self.texture_valid_mask: Optional[torch.Tensor] = None
		self.source_rgb: Optional[torch.Tensor] = None

	@staticmethod
	def _load_orientation(value: Union[STOrientation, str, Path]) -> STOrientation:
		"""Return an existing orientation unchanged, or load its XML path."""
		if isinstance(value, STOrientation):
			return value
		if isinstance(value, (str, Path)):
			path = utils.resolve_existing_path(value, Path(__file__).resolve().parent.parent)
			return STOrientation(str(path))
		raise TypeError("orientation must be an STOrientation or XML path.")

	@staticmethod
	def _load_euclidien_transform(value: Union[Lamb93_Euc_Local, str, Path]) -> Lamb93_Euc_Local:
		"""Return the local-frame object unchanged, or read its XML definition."""
		if isinstance(value, Lamb93_Euc_Local):
			return value
		if isinstance(value, (str, Path)):
			path = utils.resolve_existing_path(value, Path(__file__).resolve().parent.parent)
			return Lamb93_Euc_Local(str(path))
		raise TypeError("euclidien_transform must be a Lamb93_Euc_Local or XML path.")

	@staticmethod
	def _get_frame_center(transform: Lamb93_Euc_Local) -> np.ndarray:
		center = np.array([
			transform.euc_center_x, transform.euc_center_y, transform.euc_center_z,
		], dtype=np.float64)
		if not np.isfinite(center).all():
			raise ValueError("Local frame center must be finite.")
		return center

	def _validate_orientation(self, orientation: STOrientation) -> None:
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
		valid_intrinsics = np.isfinite([ppa["f"], ppa["cx"], ppa["cy"]]).all() and ppa["f"] > 0
		if width <= 0 or height <= 0 or not valid_intrinsics:
			raise ValueError("Orientation must have positive image dimensions and valid intrinsics.")

	def _iter_valid_points(self) -> Iterator[Tuple[torch.Tensor, np.ndarray]]:
		"""Yield flat CUDA indices and CPU float64 XYZ in bounded, row-major chunks."""
		coordinates = [self.Xg.reshape(-1), self.Yg.reshape(-1), self.Zg.reshape(-1)]
		valid = self.valid_mask.reshape(-1)
		for start in range(0, valid.numel(), self._chunk_size):
			stop = min(start + self._chunk_size, valid.numel())
			indices = torch.nonzero(valid[start:stop], as_tuple=False).flatten() + start
			if indices.numel():
				yield indices, torch.stack([coordinate[indices] for coordinate in coordinates], dim=1).cpu().numpy()

	def _transform_geometry(self) -> None:
		"""Convert valid Lambert93 nodes in place, before float32 mesh centering."""
		coordinates = [self.Xg.reshape(-1), self.Yg.reshape(-1), self.Zg.reshape(-1)]
		valid = self.valid_mask.reshape(-1)
		for indices, world in self._iter_valid_points():
			local = np.stack([
				np.asarray(axis, dtype=np.float64).reshape(-1)
				for axis in self.euclidien_transform.lamb93_to_euclidean(*world.T)
			], axis=1)
			valid[indices] = torch.as_tensor(np.isfinite(local).all(axis=1), device=self.device)
			for axis, coordinate in enumerate(coordinates):
				coordinate[indices] = torch.as_tensor(local[:, axis], device=self.device)
		if not self.valid_mask.any().item():
			raise ValueError("DSM contains no valid local geometry.")
		for coordinate in coordinates:
			coordinate[~valid] = torch.nan

	def decode_xyz(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Return independent local float64 XYZ grids, with NaNs at invalid nodes."""
		return tuple(coordinate.clone().masked_fill(~self.valid_mask, torch.nan) for coordinate in (self.Xg, self.Yg, self.Zg))

	def project_vertices_to_image(self, source_orientation: Union[STOrientation, str, Path]) -> np.ndarray:
		"""Return (H, W, 2) source pixel coordinates, with NaNs outside coverage.

		Projection uses full local coordinates, never centered mesh vertices.
		Missing frame metadata leaves same-frame consistency to the caller.
		"""
		source_orientation = self._load_orientation(source_orientation)
		self._validate_orientation(source_orientation)
		width, height = source_orientation.intrinsic.image_size
		projected = np.full((self.H * self.W, 2), np.nan, dtype=np.float64)
		for indices, points in self._iter_valid_points():
			for index, point in zip(indices.cpu().numpy(), points):
				pixel = source_orientation.LocalToImage(point, check_FOV=True)
				inside = 0 <= pixel[0] <= width - 1 and 0 <= pixel[1] <= height - 1
				if np.isfinite(pixel).all() and inside:
					projected[index] = pixel
		return projected.reshape(self.H, self.W, 2)

	@staticmethod
	def _load_source_rgb(
		source_image: Union[str, Path, np.ndarray, torch.Tensor],
		image_size: Tuple[int, int],
		image_layout: Optional[str] = None,
	) -> np.ndarray:
		"""Load RGB or infer HWC/CHW from calibrated dimensions; return HWC float32.

		Ambiguous arrays require image_layout='HWC' or 'CHW'. File images always
		load as HWC. No resizing, channel-order guessing or distortion correction.
		"""
		if image_layout not in (None, "HWC", "CHW"):
			raise ValueError("image_layout must be None, 'HWC', or 'CHW'.")
		if isinstance(source_image, (str, Path)):
			if image_layout == "CHW":
				raise ValueError("File images load as HWC; image_layout='CHW' is for memory inputs.")
			rgb = utils.read_rgb_image(Path(source_image))
			image_layout = "HWC"
		elif isinstance(source_image, torch.Tensor):
			rgb = source_image.detach().cpu().numpy()
		elif isinstance(source_image, np.ndarray):
			rgb = source_image
		else:
			raise TypeError("source_image must be a path, numpy array, or torch tensor.")
		width, height = image_size
		layouts = {"HWC": (height, width, 3), "CHW": (3, height, width)}
		if image_layout is None:
			matches = [layout for layout, shape in layouts.items() if rgb.shape == shape]
			if len(matches) == 2:
				raise ValueError("Ambiguous RGB shape; specify image_layout='HWC' or 'CHW'.")
			if not matches:
				raise ValueError(f"Source RGB must have shape {layouts['HWC']} or {layouts['CHW']}, got {rgb.shape}.")
			image_layout = matches[0]
		if rgb.shape != layouts[image_layout]:
			raise ValueError(f"Source RGB must have shape {layouts[image_layout]} for {image_layout}, got {rgb.shape}.")
		if image_layout == "CHW":
			rgb = rgb.transpose(1, 2, 0)
		is_uint8 = rgb.dtype == np.uint8
		if not is_uint8 and not np.issubdtype(rgb.dtype, np.floating):
			raise ValueError("Source RGB must be uint8 or floating point.")
		rgb = rgb.astype(np.float32)
		if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
			raise ValueError("Source RGB must be finite and within [0, 1] or [0, 255].")
		if is_uint8 or rgb.max() > 1:
			rgb /= 255.0
		return rgb

	def build_textured_mesh(
		self,
		source_image: Union[str, Path, np.ndarray, torch.Tensor],
		source_orientation: Union[STOrientation, str, Path],
		texture_sampling_mode: str = "bilinear",
		*,
		image_layout: Optional[str] = None,
	) -> Meshes:
		"""Project source UVs, attach full-resolution RGB, and replace mesh state.

		RGB and orientation each accept paths or loaded data. HWC/CHW layout is
		inferred from calibrated dimensions, with image_layout for ambiguous
		arrays. source_rgb retains normalized full-resolution HWC on CUDA;
		projected_pixels and texture_valid_mask describe vertex coverage.
		"""
		source_orientation = self._load_orientation(source_orientation)
		if texture_sampling_mode not in ("nearest", "bilinear"):
			raise ValueError("texture_sampling_mode must be 'nearest' or 'bilinear'.")
		self._validate_orientation(source_orientation)
		width, height = source_orientation.intrinsic.image_size
		rgb = self._load_source_rgb(source_image, (width, height), image_layout)
		projected = self.project_vertices_to_image(source_orientation)
		texture_valid = np.isfinite(projected).all(axis=2)
		texture_valid_tensor = torch.as_tensor(texture_valid, device=self.device)

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
		source_rgb = torch.as_tensor(rgb, device=self.device)
		verts_uvs = np.zeros((self.H, self.W, 2), dtype=np.float32)
		cols, rows = projected[texture_valid].T
		verts_uvs[..., 0][texture_valid] = cols / (width - 1) if width > 1 else 0.5
		verts_uvs[..., 1][texture_valid] = 1 - rows / (height - 1) if height > 1 else 0.5
		textures = TexturesUV(
			maps=source_rgb.unsqueeze(0),
			verts_uvs=torch.as_tensor(verts_uvs.reshape(-1, 2), device=self.device).unsqueeze(0),
			faces_uvs=faces.unsqueeze(0), sampling_mode=texture_sampling_mode,
			align_corners=True,
		)
		mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
		self.projected_pixels = projected
		self.texture_valid_mask = texture_valid_tensor
		self.render_origin_world = origin
		self.vertices = vertices
		self.faces = faces
		self.mesh = mesh
		self.source_rgb = source_rgb
		return mesh

	def render_perspective(
		self,
		camera: PerspectiveCameras,
		image_size: Tuple[int, int],
		faces_per_pixel: int = 1,
		blur_radius: float = 0.0,
		cull_backfaces: bool = False,
		bin_size: Optional[int] = None,
		max_faces_per_bin: Optional[int] = None,
		render_mode: str = "texture_only",
		enforce_cuda_outputs: bool = False,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Render a prepared camera at (height, width); see mesh_utils rendering."""
		if self.mesh is None:
			raise RuntimeError("Textured mesh is not built. Call build_textured_mesh(...) first.")
		if render_mode not in ("texture_only", "phong"):
			raise ValueError("render_mode must be 'texture_only' or 'phong'")
		if not utils.device_matches_requested(self.mesh.device, self.device):
			self.mesh = self.mesh.to(self.device)
		return utils.render_textured_mesh(
			self.mesh, camera, image_size, self.device,
			faces_per_pixel=faces_per_pixel, blur_radius=blur_radius,
			cull_backfaces=cull_backfaces, bin_size=bin_size,
			max_faces_per_bin=max_faces_per_bin, render_mode=render_mode,
			enforce_cuda_outputs=enforce_cuda_outputs,
		)

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
		R accepts (3, 3)/(1, 3, 3); T accepts (3,)/(1, 3). Camera-axis conversion
		flips image X/Y for PyTorch3D, after centering T in float64.
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
		valid_intrinsics = np.isfinite(focal + principal).all() and min(focal) > 0
		if height <= 0 or width <= 0 or not valid_intrinsics:
			raise ValueError("Camera image size and focal lengths must be positive; intrinsics must be finite.")
		return PerspectiveCameras(
			device=self.device, R=converted_rotation, T=translation,
			focal_length=(focal,), principal_point=(principal,),
			image_size=((height, width),), in_ndc=False,
		)

	def create_perspective_camera_from_orientation(
		self, target_orientation: Union[STOrientation, str, Path],
	) -> PerspectiveCameras:
		"""Read the target pose and calibration without changing the source image."""
		target_orientation = self._load_orientation(target_orientation)
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
		target_orientation: Union[STOrientation, str, Path],
		faces_per_pixel: int = 1,
		blur_radius: float = 0.0,
		cull_backfaces: bool = False,
		bin_size: Optional[int] = None,
		max_faces_per_bin: Optional[int] = None,
		render_mode: str = "texture_only",
		enforce_cuda_outputs: bool = False,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Return CUDA HWC RGB in [0, 1] and camera-axis depth with NaN background.

		Target orientation accepts an object or XML path; image_size is (W,H).
		Reuse the mesh for multiple targets; rebuild it to change the source image.
		"""
		target_orientation = self._load_orientation(target_orientation)
		camera = self.create_perspective_camera_from_orientation(target_orientation)
		width, height = target_orientation.intrinsic.image_size
		return self.render_perspective(
			camera=camera, image_size=(height, width), faces_per_pixel=faces_per_pixel,
			blur_radius=blur_radius, cull_backfaces=cull_backfaces, bin_size=bin_size,
			max_faces_per_bin=max_faces_per_bin, render_mode=render_mode,
			enforce_cuda_outputs=enforce_cuda_outputs,
		)

if __name__ == "__main__":
	ground_name = "Paris-140418_0545-301-00002_0000673"
	aerial_name = "FAUUPARIS10x00026_24551_19052014_675_3466"
	root = Path("/mast3r_ign/data/chantier_paris/preview_301/mast3r_sample_data") / ground_name
	dsm_path, vrt_path = root / f"{ground_name}_dsm_cropped.tif", root / f"{ground_name}_dsm_cropped.vrt"
	source_ori = STOrientation(root / "pva_thumbnails" / f"{aerial_name}.xml")
	source_image_path = root / "pva_thumbnails" / f"{aerial_name}.tif"
	surface = OrientedImageTexturedDSMMesh(dsm_path, source_ori.euclidien_transform, vrt_path=vrt_path)
	surface.build_textured_mesh(source_image_path, source_ori)
	target_ori = STOrientation(root / f"{ground_name}_upright_euc.xml")
	rgb, depth = surface.render_from_orientation(target_ori)

	output_dir = root / "rendered_views"
	rgb_path = output_dir / "rgb.png"
	depth_path = output_dir / "depth.tif"
	depth_preview_path = output_dir / "depth.preview.png"
	utils.save_rgb_image(rgb, rgb_path)
	utils.save_depth_image(depth, depth_path)
	utils.save_depth_preview_image(depth, depth_preview_path)
	print(f"Saved RGB: {rgb_path}")
	print(f"Saved depth: {depth_path}")
	print(f"Saved depth preview: {depth_preview_path}")
	utils.show_rendered(rgb, depth)

