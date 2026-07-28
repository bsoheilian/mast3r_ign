from pathlib import Path
from typing import Optional, Tuple, Union
from xml.etree import ElementTree
import importlib

import numpy as np
import torch
from pytorch3d.renderer import (
	AmbientLights,
	BlendParams,
	HardPhongShader,
	MeshRasterizer,
	MeshRenderer,
	PerspectiveCameras,
	RasterizationSettings,
	TexturesUV,
	look_at_view_transform,
)
from pytorch3d.structures import Meshes

try:
	from .export_to_ply import export_xyzrgb_points_to_ply,export_ply
except ImportError:
	from export_to_ply import export_xyzrgb_points_to_ply,export_ply


class mesh_from_DSM:
	"""
	Build ground coordinate grids from a DSM raster and a companion VRT file.

	The DSM TIFF provides elevation values (Z), and the VRT provides the affine
	georeferencing transform used to decode X/Y at pixel centers.
	"""

	def __init__(
		self,
		dsm_tiff_path: Union[str, Path],
		vrt_path: Union[str, Path],
		nodata_value: Optional[float] = None,
	):
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA is not available. mesh_from_DSM requires CUDA.")
		self.device = torch.device("cuda")

		project_root = Path(__file__).resolve().parent.parent
		self.dsm_tiff_path = self._resolve_existing_path(dsm_tiff_path, project_root)
		self.vrt_path = self._resolve_existing_path(vrt_path, project_root)
		self.nodata_value = nodata_value

		self._geotransform, vrt_size, vrt_nodata = self._read_vrt_metadata(self.vrt_path)
		if self.nodata_value is None and vrt_nodata is not None:
			self.nodata_value = vrt_nodata

		self.Zg = self._read_dsm_as_tensor(self.dsm_tiff_path).to(self.device)
		if vrt_size is not None and self.Zg.shape != vrt_size:
			raise ValueError(
				f"DSM shape {self.Zg.shape} does not match VRT raster size {vrt_size}."
			)

		self.Xg, self.Yg = self._decode_xy_from_geotransform(
			self.Zg.shape,
			self._geotransform,
			self.device,
		)

		self.H, self.W = self.Zg.shape
		self.valid_mask = self._build_valid_mask(self.Zg, self.nodata_value)
		self.mesh: Optional[Meshes] = None
		self.vertices: Optional[torch.Tensor] = None
		self.faces: Optional[torch.Tensor] = None
		self.verts_uvs: Optional[torch.Tensor] = None
		self.render_origin_world: Optional[torch.Tensor] = None

	@staticmethod
	def _resolve_existing_path(path_like: Union[str, Path], project_root: Path) -> Path:
		p = Path(path_like).expanduser()

		candidates = []
		if p.is_absolute():
			candidates.append(p)
		else:
			candidates.append(p)
			candidates.append(Path.cwd() / p)
			candidates.append(project_root / p)

		for candidate in candidates:
			if candidate.exists():
				return candidate.resolve()

		rendered_candidates = ", ".join(str(c) for c in candidates)
		raise FileNotFoundError(
			f"File not found for input path '{path_like}'. Checked: {rendered_candidates}"
		)

	@staticmethod
	def _read_vrt_metadata(
		vrt_path: Path,
	) -> Tuple[
		Tuple[float, float, float, float, float, float],
		Optional[Tuple[int, int]],
		Optional[float],
	]:
		tree = ElementTree.parse(vrt_path)
		root = tree.getroot()

		geotransform_node = root.find("GeoTransform")
		if geotransform_node is None or geotransform_node.text is None:
			raise ValueError(f"GeoTransform not found in VRT file: {vrt_path}")

		parts = [p.strip() for p in geotransform_node.text.split(",")]
		if len(parts) != 6:
			raise ValueError(
				f"GeoTransform must contain 6 comma-separated values, got {len(parts)} in {vrt_path}"
			)

		try:
			gt = tuple(float(v) for v in parts)
		except ValueError as exc:
			raise ValueError(f"Invalid GeoTransform values in VRT file: {vrt_path}") from exc

		raster_x_size = root.get("rasterXSize")
		raster_y_size = root.get("rasterYSize")
		raster_size: Optional[Tuple[int, int]] = None
		if raster_x_size is not None and raster_y_size is not None:
			try:
				raster_size = (int(raster_y_size), int(raster_x_size))
			except ValueError as exc:
				raise ValueError(f"Invalid raster size in VRT file: {vrt_path}") from exc

		nodata_node = root.find("VRTRasterBand/NoDataValue")
		nodata_value: Optional[float] = None
		if nodata_node is not None and nodata_node.text is not None:
			try:
				nodata_value = float(nodata_node.text.strip())
			except ValueError:
				nodata_value = None

		return gt, raster_size, nodata_value

	@staticmethod
	def _read_dsm_as_tensor(dsm_tiff_path: Path) -> torch.Tensor:
		# Prefer rasterio for GeoTIFF robustness, then fall back to tifffile.
		dsm: Optional[np.ndarray] = None

		try:
			rasterio = importlib.import_module("rasterio")

			with rasterio.open(dsm_tiff_path) as src:
				dsm = src.read(1)
		except Exception:
			pass

		if dsm is None:
			try:
				tifffile = importlib.import_module("tifffile")

				dsm = tifffile.imread(dsm_tiff_path)
				if dsm.ndim == 3:
					dsm = dsm[0]
			except Exception as exc:
				raise RuntimeError(
					"Failed to read DSM TIFF. Install rasterio or tifffile to load GeoTIFF data."
				) from exc

		if dsm.ndim != 2:
			raise ValueError(f"DSM TIFF must be a single-band 2D raster, got shape {dsm.shape}")

		return torch.from_numpy(dsm.astype(np.float64, copy=False))

	@staticmethod
	def _build_valid_mask(zg: torch.Tensor, nodata_value: Optional[float]) -> torch.Tensor:
		if nodata_value is None:
			return torch.ones_like(zg, dtype=torch.bool)
		if np.isnan(nodata_value):
			return ~torch.isnan(zg)
		return zg != float(nodata_value)

	@staticmethod
	def _decode_xy_from_geotransform(
		shape: Tuple[int, int],
		geotransform: Tuple[float, float, float, float, float, float],
		device: torch.device,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		h, w = shape
		gt0, gt1, gt2, gt3, gt4, gt5 = geotransform

		cols = torch.arange(w, dtype=torch.float64, device=device) + 0.5
		rows = torch.arange(h, dtype=torch.float64, device=device) + 0.5
		r, c = torch.meshgrid(rows, cols, indexing="ij")

		# GDAL affine model:
		# X = GT0 + c*GT1 + r*GT2
		# Y = GT3 + c*GT4 + r*GT5
		xg = gt0 + c * gt1 + r * gt2
		yg = gt3 + c * gt4 + r * gt5

		return xg, yg

	def decode_xyz(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Return decoded ground grids (Xg, Yg, Zg), all torch.float64 of shape (H, W)."""
		xg = self.Xg.clone()
		yg = self.Yg.clone()
		zg = self.Zg.clone()

		invalid = ~self.valid_mask
		if invalid.any():
			xg[invalid] = torch.nan
			yg[invalid] = torch.nan
			zg[invalid] = torch.nan

		return xg, yg, zg

	def build_textured_mesh(
		self,
		rgb_image: Union[np.ndarray, torch.Tensor],
		texture_sampling_mode: str = "bilinear",
	) -> Meshes:
		"""Create a textured triangle mesh on CUDA from DSM grids and an aligned RGB image."""
		if texture_sampling_mode not in ("nearest", "bilinear"):
			raise ValueError("texture_sampling_mode must be 'nearest' or 'bilinear'")

		if isinstance(rgb_image, np.ndarray):
			rgb = torch.from_numpy(rgb_image)
		elif isinstance(rgb_image, torch.Tensor):
			rgb = rgb_image
		else:
			raise TypeError("rgb_image must be numpy.ndarray or torch.Tensor")

		if rgb.ndim != 3 or rgb.shape[2] < 3:
			raise ValueError(f"RGB image must have shape (H, W, 3+), got {tuple(rgb.shape)}")
		if rgb.shape[0] != self.H or rgb.shape[1] != self.W:
			raise ValueError(f"RGB shape {tuple(rgb.shape[:2])} must match DSM shape {(self.H, self.W)}")

		rgb = rgb[..., :3].to(self.device)
		if rgb.dtype != torch.float32:
			rgb = rgb.float()
		if rgb.max().item() > 1.0:
			rgb = rgb / 255.0

		# Recenter geometry around a local origin to reduce float32 precision issues.
		x0 = torch.median(self.Xg)
		y0 = torch.median(self.Yg)
		z0 = torch.median(self.Zg)
		self.render_origin_world = torch.stack([x0, y0, z0], dim=0)

		x = (self.Xg - x0).float()
		y = (self.Yg - y0).float()
		z = (self.Zg - z0).float()
		self.vertices = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

		ii = torch.arange(self.H - 1, device=self.device)
		jj = torch.arange(self.W - 1, device=self.device)
		I, J = torch.meshgrid(ii, jj, indexing="ij")
		top_left = I * self.W + J
		top_right = top_left + 1
		bottom_left = top_left + self.W
		bottom_right = bottom_left + 1

		faces = torch.stack(
			[
				torch.stack([top_left, bottom_left, top_right], dim=-1),
				torch.stack([top_right, bottom_left, bottom_right], dim=-1),
			],
			dim=0,
		).reshape(-1, 3).to(dtype=torch.long)

		# Drop triangles touching invalid DSM pixels.
		node_valid = self.valid_mask.flatten()
		face_valid = node_valid[faces].all(dim=1)
		self.faces = faces[face_valid]

		xs = torch.linspace(0.0, 1.0, self.W, device=self.device)
		ys = torch.linspace(1.0, 0.0, self.H, device=self.device)
		v, u = torch.meshgrid(ys, xs, indexing="ij")
		self.verts_uvs = torch.stack([u, v], dim=-1).reshape(-1, 2)

		textures = TexturesUV(
			maps=rgb.unsqueeze(0),
			verts_uvs=self.verts_uvs.unsqueeze(0),
			faces_uvs=self.faces.unsqueeze(0),
			sampling_mode=texture_sampling_mode,
		)
		self.mesh = Meshes(verts=[self.vertices], faces=[self.faces], textures=textures)
		return self.mesh

	def create_perspective_camera(
		self,
		eye: Tuple[float, float, float],
		at: Tuple[float, float, float],
		up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
		fx_px: float = 1000.0,
		fy_px: Optional[float] = None,
		cx_px: float = 0.0,
		cy_px: float = 0.0,
		image_size: Tuple[int, int] = (1024, 1024),
	) -> PerspectiveCameras:
		"""Create a CUDA perspective camera from eye/target/up and pixel intrinsics."""
		if fy_px is None:
			fy_px = fx_px

		eye_t = torch.tensor([eye], device=self.device, dtype=torch.float32)
		at_t = torch.tensor([at], device=self.device, dtype=torch.float32)
		up_t = torch.tensor([up], device=self.device, dtype=torch.float32)
		R, T = look_at_view_transform(eye=eye_t, at=at_t, up=up_t, device=self.device)

		H, W = int(image_size[0]), int(image_size[1])
		return PerspectiveCameras(
			device=self.device,
			R=R,
			T=T,
			focal_length=((float(fx_px), float(fy_px)),),
			principal_point=((float(cx_px), float(cy_px)),),
			image_size=((H, W),),
			in_ndc=False,
		)

	def create_perspective_camera_from_rt(
		self,
		R: Union[np.ndarray, torch.Tensor],
		T: Union[np.ndarray, torch.Tensor, Tuple[float, float, float]],
		focal_length_px: Union[float, Tuple[float, float]],
		principal_point_px: Tuple[float, float],
		image_size: Tuple[int, int],
	) -> PerspectiveCameras:
		"""
		Create a CUDA perspective camera from:
		- R: image(camera)-to-world rotation
		- T: camera center in world coordinates

		The method converts these inputs to PyTorch3D world-to-view parameters.
		"""
		if isinstance(R, np.ndarray):
			R_t = torch.from_numpy(R)
		else:
			R_t = R

		if isinstance(T, np.ndarray):
			T_t = torch.from_numpy(T)
		elif isinstance(T, tuple):
			T_t = torch.tensor(T)
		else:
			T_t = T

		R_t = R_t.to(self.device, dtype=torch.float32)
		T_t = T_t.to(self.device, dtype=torch.float32)

		if R_t.ndim == 2:
			R_t = R_t.unsqueeze(0)
		if T_t.ndim == 1:
			T_t = T_t.unsqueeze(0)

		if R_t.shape != (1, 3, 3):
			raise ValueError(f"R must have shape (3, 3) or (1, 3, 3), got {tuple(R_t.shape)}")
		if T_t.shape != (1, 3):
			raise ValueError(f"T must have shape (3,) or (1, 3), got {tuple(T_t.shape)}")
		if self.render_origin_world is None:
			raise RuntimeError("Textured mesh is not built. Call build_textured_mesh(...) before creating camera.")

		# Input convention from caller:
		#   R is camera(image)->world rotation
		#   T is camera center C in world coordinates
		# PyTorch3D expects row-vector world->view transform X_cam = X_world @ R_p3d + T_p3d.
		# With this convention, convert as:
		#   R_p3d = R_cam_to_world
		#   T_p3d = -C_world @ R_cam_to_world
		R_p3d = R_t
		origin = self.render_origin_world.to(self.device, dtype=torch.float32).unsqueeze(0)
		C_local = T_t - origin
		T_p3d = -torch.bmm(C_local.unsqueeze(1), R_p3d).squeeze(1)

		if isinstance(focal_length_px, tuple):
			fx_px, fy_px = float(focal_length_px[0]), float(focal_length_px[1])
		else:
			fx_px = float(focal_length_px)
			fy_px = float(focal_length_px)

		cx_px, cy_px = float(principal_point_px[0]), float(principal_point_px[1])
		H, W = int(image_size[0]), int(image_size[1])

		return PerspectiveCameras(
			device=self.device,
			R=R_p3d,
			T=T_p3d,
			focal_length=((fx_px, fy_px),),
			principal_point=((cx_px, cy_px),),
			image_size=((H, W),),
			in_ndc=False,
		)

	def render_perspective(
		self,
		camera: PerspectiveCameras,
		image_size: Tuple[int, int],
		faces_per_pixel: int = 1,
		blur_radius: float = 0.0,
		cull_backfaces: bool = False,
		bin_size: int = 0,
		max_faces_per_bin: Optional[int] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Render textured RGB and depth map from the current mesh using a perspective camera."""
		if self.mesh is None:
			raise RuntimeError("Textured mesh is not built. Call build_textured_mesh(...) first.")

		raster = RasterizationSettings(
			image_size=(int(image_size[0]), int(image_size[1])),
			blur_radius=blur_radius,
			faces_per_pixel=faces_per_pixel,
			cull_backfaces=cull_backfaces,
			bin_size=bin_size,
			max_faces_per_bin=max_faces_per_bin,
		)
		rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster)

		lights = AmbientLights(device=self.device)
		blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
		renderer = MeshRenderer(
			rasterizer=rasterizer,
			shader=HardPhongShader(
				device=self.device,
				cameras=camera,
				lights=lights,
				blend_params=blend_params,
			),
		)

		rgba = renderer(self.mesh)[0]
		rgb = rgba[..., :3]

		fragments = rasterizer(self.mesh)
		pix_to_face = fragments.pix_to_face[0, ..., 0]
		depth = fragments.zbuf[0, ..., 0]
		depth = depth.clone()
		depth[pix_to_face < 0] = torch.nan
		depth[~torch.isfinite(depth)] = torch.nan

		return rgb, depth


def _read_rgb_image(rgb_path: Path) -> np.ndarray:
	rgb_path = mesh_from_DSM._resolve_existing_path(rgb_path, Path(__file__).resolve().parent.parent)

	try:
		pil_image_module = importlib.import_module("PIL.Image")
	except Exception as exc:
		raise RuntimeError("Failed to import PIL.Image. Install pillow to read RGB files.") from exc

	with pil_image_module.open(rgb_path) as img:
		# Make a writable copy to avoid torch.from_numpy non-writable warning.
		rgb = np.array(img.convert("RGB"), copy=True)

	if rgb.ndim != 3 or rgb.shape[2] != 3:
		raise ValueError(f"RGB image must be (H, W, 3), got shape {rgb.shape}")

	return rgb


def _finite_min_max(t: torch.Tensor) -> Tuple[float, float]:
	finite = t[torch.isfinite(t)]
	if finite.numel() == 0:
		return float("nan"), float("nan")
	return float(finite.min().item()), float(finite.max().item())


def _save_rgb_image(rgb: torch.Tensor, out_path: Union[str, Path]) -> None:
	out_path = Path(out_path)
	rgb_np = rgb.detach().clamp(0.0, 1.0).cpu().numpy()
	rgb_u8 = (rgb_np * 255.0).round().astype(np.uint8)

	try:
		pil_image_module = importlib.import_module("PIL.Image")
	except Exception as exc:
		raise RuntimeError("Failed to import PIL.Image. Install pillow to save RGB images.") from exc

	out_path.parent.mkdir(parents=True, exist_ok=True)
	img = pil_image_module.fromarray(rgb_u8, mode="RGB")
	img.save(out_path)


def _save_depth_image(depth: torch.Tensor, out_path: Union[str, Path]) -> None:
	out_path = Path(out_path)
	depth_np = depth.detach().cpu().numpy().astype(np.float32)

	try:
		pil_image_module = importlib.import_module("PIL.Image")
	except Exception as exc:
		raise RuntimeError("Failed to import PIL.Image. Install pillow to save depth images.") from exc

	out_path.parent.mkdir(parents=True, exist_ok=True)
	depth_img = pil_image_module.fromarray(depth_np, mode="F")
	depth_img.save(out_path)


def _save_depth_preview_image(depth: torch.Tensor, out_path: Union[str, Path]) -> None:
	"""Save a normalized 8-bit depth preview for quick visual checks."""
	out_path = Path(out_path)
	depth_np = depth.detach().cpu().numpy().astype(np.float32)
	valid = np.isfinite(depth_np)

	preview = np.zeros_like(depth_np, dtype=np.uint8)
	if np.any(valid):
		dmin = float(np.nanmin(depth_np[valid]))
		dmax = float(np.nanmax(depth_np[valid]))
		if dmax > dmin:
			norm = (depth_np[valid] - dmin) / (dmax - dmin)
			preview[valid] = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)

	try:
		pil_image_module = importlib.import_module("PIL.Image")
	except Exception as exc:
		raise RuntimeError("Failed to import PIL.Image. Install pillow to save depth preview images.") from exc

	out_path.parent.mkdir(parents=True, exist_ok=True)
	img = pil_image_module.fromarray(preview, mode="L")
	img.save(out_path)


def _show_rendered(rgb: torch.Tensor, depth: torch.Tensor) -> None:
	try:
		plt = importlib.import_module("matplotlib.pyplot")
	except Exception as exc:
		raise RuntimeError("Failed to import matplotlib.pyplot. Install matplotlib to show renders.") from exc

	rgb_np = rgb.detach().clamp(0.0, 1.0).cpu().numpy()
	depth_np = depth.detach().cpu().numpy()

	plt.figure(figsize=(14, 6))
	plt.subplot(1, 2, 1)
	plt.imshow(rgb_np)
	plt.title("Rendered RGB")
	plt.axis("off")

	plt.subplot(1, 2, 2)
	plt.imshow(depth_np, cmap="viridis")
	plt.title("Rendered Depth")
	plt.axis("off")
	plt.tight_layout()
	plt.show()


def _select_best_camera_convention(
	mesh: "mesh_from_DSM",
	rotation_matrix: Union[np.ndarray, torch.Tensor],
	translation_vector: Union[np.ndarray, torch.Tensor, Tuple[float, float, float]],
	focal_length_px: Union[float, Tuple[float, float]],
	principal_point_px: Tuple[float, float],
	image_size: Tuple[int, int],
) -> Tuple[PerspectiveCameras, torch.Tensor, torch.Tensor, str, int, int]:
	"""Try several plausible extrinsic conventions and keep the one with most visible depth pixels."""
	R_in = np.asarray(rotation_matrix, dtype=np.float32).reshape(3, 3)
	C_in = np.asarray(translation_vector, dtype=np.float32).reshape(3)

	axis_flips = [
		("no_axis_flip", np.eye(3, dtype=np.float32)),
		("flip_yz", np.diag([1.0, -1.0, -1.0]).astype(np.float32)),
		("flip_xz", np.diag([-1.0, 1.0, -1.0]).astype(np.float32)),
		("flip_xy", np.diag([-1.0, -1.0, 1.0]).astype(np.float32)),
	]
	rot_variants = [
		("R_img2world", R_in),
		("R_world2img", R_in.T),
	]

	best = None
	total_depth = int(image_size[0] * image_size[1])

	for rot_name, R_base in rot_variants:
		for flip_name, F in axis_flips:
			# Apply local camera-axis remapping before conversion to world->view.
			R_try = R_base @ F
			cam_try = mesh.create_perspective_camera_from_rt(
				R=R_try,
				T=C_in,
				focal_length_px=focal_length_px,
				principal_point_px=principal_point_px,
				image_size=image_size,
			)
			rgb_try, depth_try = mesh.render_perspective(camera=cam_try, image_size=image_size)
			valid = int(torch.isfinite(depth_try).sum().item())
			label = f"{rot_name}+{flip_name}"
			print(f"Camera candidate {label}: valid depth {valid}/{total_depth}")

			if best is None or valid > best[4]:
				best = (cam_try, rgb_try, depth_try, label, valid, total_depth)

			# Early stop when a convention yields full visibility.
			if valid == total_depth:
				return best

	# best is guaranteed set because candidate list is non-empty.
	return best


def main(
	dsm_tiff_path: Union[str, Path],
	vrt_path: Union[str, Path],
	rgb_path: Union[str, Path],
	rotation_matrix: Union[np.ndarray, torch.Tensor],
	translation_vector: Union[np.ndarray, torch.Tensor, Tuple[float, float, float]],
	focal_length_px: Union[float, Tuple[float, float]],
	principal_point_px: Tuple[float, float],
	image_size: Tuple[int, int],
	points_ply_path: Union[str, Path] = "points_rgb.ply",
	rendered_rgb_path: Union[str, Path] = "rendered_rgb.png",
	rendered_depth_path: Union[str, Path] = "rendered_depth.tif",
	show_rendered_images: bool = True,
	nodata_value: Optional[float] = None,
) -> mesh_from_DSM:
	"""
	Initialize DSM mesh data on CUDA, build textured mesh, and export colored point cloud.
	Then render perspective RGB/depth from explicit camera parameters, save them, and optionally show them.

	Returns:
		The initialized and textured mesh_from_DSM instance.
	"""

	mesh = mesh_from_DSM(
		dsm_tiff_path=dsm_tiff_path,
		vrt_path=vrt_path,
		nodata_value=nodata_value,
	)
	xg, yg, zg = mesh.decode_xyz()

	valid_count = int(mesh.valid_mask.sum().item())
	total_count = int(mesh.valid_mask.numel())
	x_min, x_max = _finite_min_max(xg)
	y_min, y_max = _finite_min_max(yg)
	z_min, z_max = _finite_min_max(zg)

	print(f"Computation device: {mesh.device} | Xg device: {xg.device} | Zg device: {zg.device}")
	print(f"Initialized mesh_from_DSM: H={mesh.H}, W={mesh.W}")
	print(f"Valid pixels: {valid_count}/{total_count}")
	print(
		"X range: [{:.6f}, {:.6f}] | Y range: [{:.6f}, {:.6f}] | Z range: [{:.6f}, {:.6f}]".format(
			x_min,
			x_max,
			y_min,
			y_max,
			z_min,
			z_max,
		)
	)

	rgb = _read_rgb_image(Path(rgb_path))
	if rgb.shape[:2] != (mesh.H, mesh.W):
		raise ValueError(
			f"RGB shape {rgb.shape[:2]} does not match DSM shape {(mesh.H, mesh.W)}. "
			"Provide an RGB orthophoto aligned to the DSM grid."
		)

	print(f"Loaded RGB overlay: shape={rgb.shape}, dtype={rgb.dtype}")
	rgb_cuda = torch.from_numpy(rgb).to(mesh.device)
	mesh.build_textured_mesh(rgb_cuda)
	print(f"Built textured mesh: verts={mesh.vertices.shape[0]}, faces={mesh.faces.shape[0]}")
	if mesh.render_origin_world is not None:
		print(f"Render origin (world): {mesh.render_origin_world.detach().cpu().numpy()}")
	export_ply(mesh, "./ign_samples/dsm_ortho_2_img/output/mesh.ply")
	export_xyzrgb_points_to_ply(xg, yg, zg, rgb_cuda, path=str(points_ply_path))
	print(f"Exported colored point cloud to: {points_ply_path}")

	camera, render_rgb, render_depth, selected_label, valid_depth, total_depth = _select_best_camera_convention(
		mesh=mesh,
		rotation_matrix=rotation_matrix,
		translation_vector=translation_vector,
		focal_length_px=focal_length_px,
		principal_point_px=principal_point_px,
		image_size=image_size,
	)

	print("Perspective camera parameters:")
	print(f"Input R (image->world):\n{np.asarray(rotation_matrix)}")
	print(f"Input camera center C in world: {np.asarray(translation_vector)}")
	print(f"Selected camera convention: {selected_label}")
	print(f"Converted R for PyTorch3D (world->view row form):\n{camera.R[0].detach().cpu().numpy()}")
	print(f"Converted T for PyTorch3D (world->view row form): {camera.T[0].detach().cpu().numpy()}")
	print(f"Focal length (px): {camera.focal_length[0].detach().cpu().numpy()}")
	print(f"Principal point (px): {camera.principal_point[0].detach().cpu().numpy()}")
	print(f"Image size (H, W): {image_size}")
	print(f"Rendered valid depth pixels: {valid_depth}/{total_depth}")
	print(
		"Rendered RGB range: [{:.6f}, {:.6f}]".format(
			float(render_rgb.min().item()),
			float(render_rgb.max().item()),
		)
	)
	if valid_depth > 0:
		depth_valid = render_depth[torch.isfinite(render_depth)]
		print(
			"Rendered depth range (valid): [{:.6f}, {:.6f}]".format(
				float(depth_valid.min().item()),
				float(depth_valid.max().item()),
			)
		)

	_save_rgb_image(render_rgb, rendered_rgb_path)
	_save_depth_image(render_depth, rendered_depth_path)
	depth_preview_path = Path(rendered_depth_path).with_suffix(".preview.png")
	_save_depth_preview_image(render_depth, depth_preview_path)
	print(f"Saved perspective RGB render to: {rendered_rgb_path}")
	print(f"Saved perspective depth render to: {rendered_depth_path}")
	print(f"Saved normalized depth preview to: {depth_preview_path}")

	if show_rendered_images:
		_show_rendered(render_rgb, render_depth)

	return mesh


if __name__ == "__main__":
	# raise RuntimeError(
	# 	"This module no longer exposes a CLI entrypoint. Import and call main(...) with explicit parameters."
	# )




    import numpy as np
        
    # R.txt should contain 9 float values (3x3), comma or whitespace separated.
    # T.txt should contain 3 float values, comma or whitespace separated.
    R = np.loadtxt("./ign_samples/dsm_ortho_2_img/R.txt", dtype=np.float32)
    T = np.loadtxt("./ign_samples/dsm_ortho_2_img/T.txt", dtype=np.float32)
    
    # If needed, enforce expected shapes.
    R = R.reshape(3, 3)
    T = T.reshape(3)
    
    mesh = main(
        dsm_tiff_path=  "./ign_samples/dsm_ortho_2_img/rge_dsm_corel.tif",
        vrt_path=       "./ign_samples/dsm_ortho_2_img/rge_dsm_corel.vrt",
        rgb_path=       "./ign_samples/dsm_ortho_2_img/rge_ortho.tif",
        rotation_matrix=R,
        translation_vector=T,
        focal_length_px=1396.939,
        principal_point_px=(965.995, 539.776),
        image_size=(1080, 1920),
        points_ply_path=        "./ign_samples/dsm_ortho_2_img/output/points_rgb.ply",
        rendered_rgb_path=      "./ign_samples/dsm_ortho_2_img/output/rgb.png",
        rendered_depth_path=    "./ign_samples/dsm_ortho_2_img/output/depth.tif",
        show_rendered_images=True,
        nodata_value=None,
)
