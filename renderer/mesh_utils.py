"""Stateless raster/image I/O and PyTorch3D textured-mesh rendering.

Raster coordinates describe pixel centers in the input world frame. Rendering
accepts a mesh and camera already expressed in the same frame; image_size is
(height, width). RGB is HWC in [0, 1], depth is camera-axis Z with NaN background.
Image/display dependencies are loaded on demand. No orientation or PLY export
code is imported here, so these functions can serve other mesh classes.
"""

import importlib
from pathlib import Path
from typing import Optional, Tuple, Union
from xml.etree import ElementTree

import numpy as np
import torch
from pytorch3d.renderer import (
	AmbientLights, BlendParams, HardPhongShader, MeshRasterizer, MeshRenderer,
	PerspectiveCameras, RasterizationSettings,
)
from pytorch3d.structures import Meshes


def resolve_existing_path(path_like: Union[str, Path], project_root: Path) -> Path:
	"""Resolve an input relative to the working directory or project root."""
	path = Path(path_like).expanduser()
	candidates = [path] if path.is_absolute() else [path, Path.cwd() / path, project_root / path]
	for candidate in candidates:
		if candidate.exists():
			return candidate.resolve()
	raise FileNotFoundError(
		f"File not found for input path '{path_like}'. Checked: {', '.join(str(path) for path in candidates)}"
	)


def device_matches_requested(device: torch.device, requested: torch.device) -> bool:
	"""Treat cuda and cuda:0 as compatible when no index was requested."""
	if device.type != requested.type:
		return False
	if requested.type != "cuda":
		return device == requested
	return requested.index is None or device.index == requested.index


def read_vrt_metadata(
	vrt_path: Path,
) -> Tuple[Tuple[float, float, float, float, float, float], Optional[Tuple[int, int]], Optional[float]]:
	"""Return GDAL geotransform, optional (height, width), and band-one nodata."""
	root = ElementTree.parse(vrt_path).getroot()
	gt_node = root.find("GeoTransform")
	if gt_node is None or gt_node.text is None:
		raise ValueError(f"GeoTransform not found in VRT file: {vrt_path}")
	parts = [part.strip() for part in gt_node.text.split(",")]
	if len(parts) != 6:
		raise ValueError(f"GeoTransform must contain 6 values, got {len(parts)} in {vrt_path}")
	geotransform = tuple(float(value) for value in parts)
	width, height = root.get("rasterXSize"), root.get("rasterYSize")
	raster_size = (int(height), int(width)) if width is not None and height is not None else None
	nodata = None
	nodata_node = root.find("VRTRasterBand/NoDataValue")
	if nodata_node is not None and nodata_node.text is not None:
		try:
			nodata = float(nodata_node.text.strip())
		except ValueError:
			pass
	return geotransform, raster_size, nodata


def read_dsm_as_tensor(dsm_tiff_path: Path) -> torch.Tensor:
	"""Read a single-band DSM as CPU float64, with a tifffile fallback."""
	dsm = None
	try:
		rasterio = importlib.import_module("rasterio")
		with rasterio.open(dsm_tiff_path) as source:
			dsm = source.read(1)
	except Exception:
		pass
	if dsm is None:
		try:
			tifffile = importlib.import_module("tifffile")
			dsm = tifffile.imread(dsm_tiff_path)
			if dsm.ndim == 3:
				dsm = dsm[0]
		except Exception as exc:
			raise RuntimeError("Failed to read DSM TIFF. Install rasterio or tifffile.") from exc
	if dsm.ndim != 2:
		raise ValueError(f"DSM TIFF must be single-band 2D, got shape {dsm.shape}")
	return torch.from_numpy(dsm.astype(np.float64, copy=False))


def load_surface(
	surface: Union[str, Path, np.ndarray, torch.Tensor],
	*,
	profile: Optional[dict] = None,
	vrt_path: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, dict]:
	"""Load elevations and metadata, or normalize an in-memory raster/profile.

	Paths may identify GeoTIFFs or VRTs; band one is read. Memory inputs must
	be (H,W) or (1,H,W), with a profile containing an Affine transform. A VRT
	metadata override is allowed only for file inputs. Declared CRS must be
	Lambert93; absent CRS leaves that responsibility to the caller. Returned
	float64 elevations and profile are independent of caller-owned inputs.
	"""
	import rasterio
	from affine import Affine

	project_root = Path(__file__).resolve().parent.parent
	is_path = isinstance(surface, (str, Path))
	if is_path:
		if profile is not None:
			raise ValueError("profile is only accepted for in-memory surfaces.")
		path = resolve_existing_path(surface, project_root)
		with rasterio.open(path) as source:
			data, metadata = source.read(1), source.profile.copy()
		if vrt_path is not None:
			override_path = resolve_existing_path(vrt_path, project_root)
			geotransform, size, nodata = read_vrt_metadata(override_path)
			if size is not None and data.shape != size:
				raise ValueError(f"Surface shape {data.shape} does not match VRT raster size {size}.")
			with rasterio.open(override_path) as source:
				metadata["crs"] = source.crs
			metadata.update(transform=Affine.from_gdal(*geotransform), nodata=nodata)
	else:
		if not isinstance(surface, (np.ndarray, torch.Tensor)):
			raise TypeError("surface must be a raster path, numpy array, or torch tensor.")
		if vrt_path is not None:
			raise ValueError("vrt_path is only accepted for file surfaces; use profile for memory inputs.")
		if profile is None:
			raise ValueError("An in-memory surface requires a profile containing its transform.")
		data, metadata = surface, dict(profile)
		if data.ndim == 3 and data.shape[0] == 1:
			data = data[0]
	if data.ndim != 2 or min(data.shape) == 0:
		raise ValueError("Surface elevations must have shape (H,W) or (1,H,W).")
	height, width = data.shape
	for key, expected in (("height", height), ("width", width)):
		if key in metadata and metadata[key] != expected:
			raise ValueError(f"Profile {key} does not match surface data: expected {expected}.")
	if not is_path and metadata.get("count", 1) != 1:
		raise ValueError("An in-memory elevation profile must describe one band.")
	transform = metadata.get("transform")
	if not isinstance(transform, Affine) or not np.isfinite(transform.to_gdal()).all():
		raise ValueError("Surface profile must contain a finite Affine transform.")
	crs = metadata.get("crs")
	if crs is not None and rasterio.crs.CRS.from_user_input(crs).to_epsg() != 2154:
		raise ValueError("Surface CRS must be Lambert93 (EPSG:2154); reproject before rendering.")
	metadata.update(height=height, width=width, count=1)
	if isinstance(data, np.ndarray):
		data = np.array(data, dtype=np.float64, copy=True, order="C")
		elevations = torch.from_numpy(data)
	else:
		elevations = data.detach().to(dtype=torch.float64).clone()
	return elevations, metadata


def build_valid_mask(zg: torch.Tensor, nodata_value: Optional[float]) -> torch.Tensor:
	"""Mask explicit nodata; callers separately reject other nonfinite values."""
	if nodata_value is None:
		return torch.ones_like(zg, dtype=torch.bool)
	return ~torch.isnan(zg) if np.isnan(nodata_value) else zg != float(nodata_value)


def decode_xy_from_geotransform(
	shape: Tuple[int, int],
	geotransform: Tuple[float, float, float, float, float, float],
	device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Decode float64 pixel-center world XY, including rotated raster grids."""
	height, width = shape
	gt0, gt1, gt2, gt3, gt4, gt5 = geotransform
	cols = torch.arange(width, dtype=torch.float64, device=device) + 0.5
	rows = torch.arange(height, dtype=torch.float64, device=device) + 0.5
	row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
	return gt0 + col_grid * gt1 + row_grid * gt2, gt3 + col_grid * gt4 + row_grid * gt5


def render_textured_mesh(
	mesh: Meshes,
	camera: PerspectiveCameras,
	image_size: Tuple[int, int],
	device: torch.device,
	faces_per_pixel: int = 1,
	blur_radius: float = 0.0,
	cull_backfaces: bool = False,
	bin_size: Optional[int] = None,
	max_faces_per_bin: Optional[int] = None,
	render_mode: str = "texture_only",
	enforce_cuda_outputs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Render nearest-surface RGB/depth, with optional ambient Phong shading.

	Background RGB is black and depth is NaN. Inputs are moved to device as
	needed without changing caller-owned references. No camera conversion or
	source-image visibility calculation is performed.
	"""
	if render_mode not in ("texture_only", "phong"):
		raise ValueError("render_mode must be 'texture_only' or 'phong'")
	if not device_matches_requested(mesh.device, device):
		mesh = mesh.to(device)
	if not device_matches_requested(camera.device, device):
		camera = camera.to(device)
	raster = RasterizationSettings(
		image_size=(int(image_size[0]), int(image_size[1])), blur_radius=blur_radius,
		faces_per_pixel=faces_per_pixel, cull_backfaces=cull_backfaces,
		bin_size=bin_size, max_faces_per_bin=max_faces_per_bin,
	)
	rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster)
	fragments = rasterizer(mesh)
	pix_to_face = fragments.pix_to_face[0, ..., 0]
	if render_mode == "texture_only":
		rgb = mesh.sample_textures(fragments)[0, ..., 0, :].clone()
		rgb[pix_to_face < 0] = 0.0
	else:
		renderer = MeshRenderer(
			rasterizer=rasterizer,
			shader=HardPhongShader(
				device=device, cameras=camera, lights=AmbientLights(device=device),
				blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
			),
		)
		rgb = renderer(mesh)[0, ..., :3]
	depth = fragments.zbuf[0, ..., 0].clone()
	depth[pix_to_face < 0] = torch.nan
	depth[~torch.isfinite(depth)] = torch.nan
	print(
		"Render devices: "
		f"mesh={mesh.device}, camera={camera.device}, "
		f"fragments={fragments.zbuf.device}, rgb={rgb.device}, depth={depth.device}"
	)
	if not device_matches_requested(rgb.device, device) or not device_matches_requested(depth.device, device):
		message = (
			"Render outputs are not on requested CUDA device. "
			f"rgb={rgb.device}, depth={depth.device}, requested={device}. "
			"This usually means a CPU fallback in the current PyTorch3D build/runtime."
		)
		if enforce_cuda_outputs:
			raise RuntimeError(message)
		print(f"[WARN] {message}")
	return rgb, depth


def read_rgb_image(rgb_path: Path) -> np.ndarray:
	"""Read source RGB as a uint8 HWC array without geospatial resampling."""
	rgb_path = resolve_existing_path(rgb_path, Path(__file__).resolve().parent.parent)
	image_module = importlib.import_module("PIL.Image")
	with image_module.open(rgb_path) as image:
		rgb = np.array(image.convert("RGB"), copy=True)
	if rgb.ndim != 3 or rgb.shape[2] != 3:
		raise ValueError(f"RGB image must be (H, W, 3), got shape {rgb.shape}")
	return rgb


def save_rgb_image(rgb: torch.Tensor, out_path: Union[str, Path]) -> None:
	"""Save normalized HWC RGB as an 8-bit image, creating its directory."""
	out_path = Path(out_path)
	rgb_u8 = (rgb.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
	image_module = importlib.import_module("PIL.Image")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	image_module.fromarray(rgb_u8, mode="RGB").save(out_path)


def save_depth_image(depth: torch.Tensor, out_path: Union[str, Path]) -> None:
	"""Save float32 depth (including NaNs); use a floating-point format like TIFF."""
	out_path = Path(out_path)
	depth_np = depth.detach().cpu().numpy().astype(np.float32)
	image_module = importlib.import_module("PIL.Image")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	image_module.fromarray(depth_np, mode="F").save(out_path)


def save_depth_preview_image(depth: torch.Tensor, out_path: Union[str, Path]) -> None:
	"""Save a min/max-scaled 8-bit preview, with invalid depth rendered black."""
	out_path = Path(out_path)
	depth_np = depth.detach().cpu().numpy().astype(np.float32)
	valid = np.isfinite(depth_np)
	preview = np.zeros_like(depth_np, dtype=np.uint8)
	if np.any(valid):
		minimum, maximum = float(depth_np[valid].min()), float(depth_np[valid].max())
		if maximum > minimum:
			normalized = (depth_np[valid] - minimum) / (maximum - minimum)
			preview[valid] = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
	image_module = importlib.import_module("PIL.Image")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	image_module.fromarray(preview, mode="L").save(out_path)


def show_rendered(rgb: torch.Tensor, depth: torch.Tensor) -> None:
	"""Display RGB and camera-axis depth side by side in a matplotlib window."""
	plt = importlib.import_module("matplotlib.pyplot")
	plt.figure(figsize=(14, 6))
	plt.subplot(1, 2, 1)
	plt.imshow(rgb.detach().clamp(0.0, 1.0).cpu().numpy())
	plt.title("Rendered RGB")
	plt.axis("off")
	plt.subplot(1, 2, 2)
	plt.imshow(depth.detach().cpu().numpy(), cmap="viridis")
	plt.title("Rendered Depth")
	plt.axis("off")
	plt.tight_layout()
	plt.show()