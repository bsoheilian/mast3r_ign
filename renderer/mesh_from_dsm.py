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
)
from pytorch3d.structures import Meshes

try:
	from .export_to_ply import export_ply, export_xyzrgb_points_to_ply
except ImportError:
	from export_to_ply import export_ply, export_xyzrgb_points_to_ply


class mesh_from_DSM:
	"""
	Decode DSM + VRT into Xg/Yg/Zg, build a textured mesh, and render perspective RGB/depth.

	Fixed camera convention:
	1. Input R is image(camera)-to-world rotation.
	2. Input T is camera center in world coordinates.
	3. A camera-axis flip `flip_xy = diag([-1, -1, 1])` is applied.
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
			raise ValueError(f"DSM shape {self.Zg.shape} does not match VRT raster size {vrt_size}.")

		self.Xg, self.Yg = self._decode_xy_from_geotransform(self.Zg.shape, self._geotransform, self.device)
		self.H, self.W = self.Zg.shape
		self.valid_mask = self._build_valid_mask(self.Zg, self.nodata_value)

		self.mesh: Optional[Meshes] = None
		self.vertices: Optional[torch.Tensor] = None
		self.faces: Optional[torch.Tensor] = None
		self.render_origin_world: Optional[torch.Tensor] = None

	@staticmethod
	def _resolve_existing_path(path_like: Union[str, Path], project_root: Path) -> Path:
		p = Path(path_like).expanduser()
		candidates = [p] if p.is_absolute() else [p, Path.cwd() / p, project_root / p]
		for c in candidates:
			if c.exists():
				return c.resolve()
		raise FileNotFoundError(f"File not found for input path '{path_like}'. Checked: {', '.join(str(x) for x in candidates)}")

	@staticmethod
	def _device_matches_requested(device: torch.device, requested: torch.device) -> bool:
		"""Treat cuda and cuda:0 as compatible when requested has no explicit index."""
		if device.type != requested.type:
			return False
		if requested.type != "cuda":
			return device == requested
		if requested.index is None:
			return device.type == "cuda"
		return device.index == requested.index

	@staticmethod
	def _read_vrt_metadata(
		vrt_path: Path,
	) -> Tuple[Tuple[float, float, float, float, float, float], Optional[Tuple[int, int]], Optional[float]]:
		tree = ElementTree.parse(vrt_path)
		root = tree.getroot()

		gt_node = root.find("GeoTransform")
		if gt_node is None or gt_node.text is None:
			raise ValueError(f"GeoTransform not found in VRT file: {vrt_path}")
		parts = [p.strip() for p in gt_node.text.split(",")]
		if len(parts) != 6:
			raise ValueError(f"GeoTransform must contain 6 values, got {len(parts)} in {vrt_path}")
		gt = tuple(float(v) for v in parts)

		rx = root.get("rasterXSize")
		ry = root.get("rasterYSize")
		raster_size = (int(ry), int(rx)) if rx is not None and ry is not None else None

		nodata = None
		nodata_node = root.find("VRTRasterBand/NoDataValue")
		if nodata_node is not None and nodata_node.text is not None:
			try:
				nodata = float(nodata_node.text.strip())
			except ValueError:
				nodata = None

		return gt, raster_size, nodata

	@staticmethod
	def _read_dsm_as_tensor(dsm_tiff_path: Path) -> torch.Tensor:
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
				raise RuntimeError("Failed to read DSM TIFF. Install rasterio or tifffile.") from exc
		if dsm.ndim != 2:
			raise ValueError(f"DSM TIFF must be single-band 2D, got shape {dsm.shape}")
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
		xg = gt0 + c * gt1 + r * gt2
		yg = gt3 + c * gt4 + r * gt5
		return xg, yg

	def decode_xyz(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		xg, yg, zg = self.Xg.clone(), self.Yg.clone(), self.Zg.clone()
		invalid = ~self.valid_mask
		if invalid.any():
			xg[invalid] = torch.nan
			yg[invalid] = torch.nan
			zg[invalid] = torch.nan
		return xg, yg, zg

	def build_textured_mesh(self, rgb_image: Union[np.ndarray, torch.Tensor], texture_sampling_mode: str = "bilinear") -> Meshes:
		if texture_sampling_mode not in ("nearest", "bilinear"):
			raise ValueError("texture_sampling_mode must be 'nearest' or 'bilinear'")
		rgb = torch.from_numpy(rgb_image) if isinstance(rgb_image, np.ndarray) else rgb_image
		if not isinstance(rgb, torch.Tensor):
			raise TypeError("rgb_image must be numpy.ndarray or torch.Tensor")
		if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.shape[0] != self.H or rgb.shape[1] != self.W:
			raise ValueError(f"RGB must be (H, W, 3+) with H,W={(self.H, self.W)}, got {tuple(rgb.shape)}")

		rgb = rgb[..., :3].to(self.device)
		if rgb.dtype != torch.float32:
			rgb = rgb.float()
		if rgb.max().item() > 1.0:
			rgb = rgb / 255.0

		x0, y0, z0 = torch.median(self.Xg), torch.median(self.Yg), torch.median(self.Zg)
		self.render_origin_world = torch.stack([x0, y0, z0], dim=0)
		x = (self.Xg - x0).float()
		y = (self.Yg - y0).float()
		z = (self.Zg - z0).float()
		self.vertices = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

		ii = torch.arange(self.H - 1, device=self.device)
		jj = torch.arange(self.W - 1, device=self.device)
		I, J = torch.meshgrid(ii, jj, indexing="ij")
		tl = I * self.W + J
		tr = tl + 1
		bl = tl + self.W
		br = bl + 1
		faces = torch.stack(
			[
				torch.stack([tl, bl, tr], dim=-1),
				torch.stack([tr, bl, br], dim=-1),
			],
			dim=0,
		).reshape(-1, 3).to(dtype=torch.long)

		node_valid = self.valid_mask.flatten()
		self.faces = faces[node_valid[faces].all(dim=1)]

		xs = torch.linspace(0.0, 1.0, self.W, device=self.device)
		ys = torch.linspace(1.0, 0.0, self.H, device=self.device)
		v, u = torch.meshgrid(ys, xs, indexing="ij")
		verts_uvs = torch.stack([u, v], dim=-1).reshape(-1, 2)

		textures = TexturesUV(
			maps=rgb.unsqueeze(0),
			verts_uvs=verts_uvs.unsqueeze(0),
			faces_uvs=self.faces.unsqueeze(0),
			sampling_mode=texture_sampling_mode,
		)
		self.mesh = Meshes(verts=[self.vertices], faces=[self.faces], textures=textures)
		if not self._device_matches_requested(self.mesh.device, self.device):
			self.mesh = self.mesh.to(self.device)
		return self.mesh

	def create_perspective_camera_from_rt(
		self,
		R: Union[np.ndarray, torch.Tensor],
		T: Union[np.ndarray, torch.Tensor, Tuple[float, float, float]],
		focal_length_px: Union[float, Tuple[float, float]],
		principal_point_px: Tuple[float, float],
		image_size: Tuple[int, int],
	) -> PerspectiveCameras:
		"""Create camera from image->world R and camera center T with fixed R_img2world+flip_xy convention."""
		if self.render_origin_world is None:
			raise RuntimeError("Call build_textured_mesh(...) before creating camera.")

		R_t = torch.from_numpy(R) if isinstance(R, np.ndarray) else R
		T_t = torch.from_numpy(T) if isinstance(T, np.ndarray) else (torch.tensor(T) if isinstance(T, tuple) else T)
		R_t = R_t.to(self.device, dtype=torch.float32)
		T_t = T_t.to(self.device, dtype=torch.float32)
		if R_t.ndim == 2:
			R_t = R_t.unsqueeze(0)
		if T_t.ndim == 1:
			T_t = T_t.unsqueeze(0)
		if R_t.shape != (1, 3, 3) or T_t.shape != (1, 3):
			raise ValueError(f"Expected R (1,3,3) and T (1,3), got {tuple(R_t.shape)} and {tuple(T_t.shape)}")

		flip_xy = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=self.device, dtype=torch.float32)).unsqueeze(0)
		R_conv = torch.bmm(R_t, flip_xy)

		origin = self.render_origin_world.to(self.device, dtype=torch.float32).unsqueeze(0)
		C_local = T_t - origin
		T_conv = -torch.bmm(C_local.unsqueeze(1), R_conv).squeeze(1)

		if isinstance(focal_length_px, tuple):
			fx_px, fy_px = float(focal_length_px[0]), float(focal_length_px[1])
		else:
			fx_px = float(focal_length_px)
			fy_px = float(focal_length_px)
		cx_px, cy_px = float(principal_point_px[0]), float(principal_point_px[1])
		H, W = int(image_size[0]), int(image_size[1])

		return PerspectiveCameras(
			device=self.device,
			R=R_conv,
			T=T_conv,
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
		bin_size: Optional[int] = None,
		max_faces_per_bin: Optional[int] = None,
		render_mode: str = "texture_only",
		enforce_cuda_outputs: bool = False,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		if self.mesh is None:
			raise RuntimeError("Textured mesh is not built. Call build_textured_mesh(...) first.")
		if render_mode not in ("texture_only", "phong"):
			raise ValueError("render_mode must be 'texture_only' or 'phong'")
		if not self._device_matches_requested(self.mesh.device, self.device):
			self.mesh = self.mesh.to(self.device)
		if not self._device_matches_requested(camera.device, self.device):
			camera = camera.to(self.device)

		raster = RasterizationSettings(
			image_size=(int(image_size[0]), int(image_size[1])),
			blur_radius=blur_radius,
			faces_per_pixel=faces_per_pixel,
			cull_backfaces=cull_backfaces,
			bin_size=bin_size,
			max_faces_per_bin=max_faces_per_bin,
		)
		rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster)
		fragments = rasterizer(self.mesh)
		pix_to_face = fragments.pix_to_face[0, ..., 0]

		if render_mode == "texture_only":
			# Fast path: sample texture directly from rasterized fragments and skip shading.
			texels = self.mesh.sample_textures(fragments)[0, ..., 0, :]
			rgb = texels.clone()
			rgb[pix_to_face < 0] = 0.0
		else:
			lights = AmbientLights(device=self.device)
			blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
			renderer = MeshRenderer(
				rasterizer=rasterizer,
				shader=HardPhongShader(device=self.device, cameras=camera, lights=lights, blend_params=blend_params),
			)
			rgba = renderer(self.mesh)[0]
			rgb = rgba[..., :3]

		depth = fragments.zbuf[0, ..., 0].clone()
		depth[pix_to_face < 0] = torch.nan
		depth[~torch.isfinite(depth)] = torch.nan
		print(
			"Render devices: "
			f"mesh={self.mesh.device}, camera={camera.device}, "
			f"fragments={fragments.zbuf.device}, rgb={rgb.device}, depth={depth.device}"
		)
		if (
			not self._device_matches_requested(rgb.device, self.device)
			or not self._device_matches_requested(depth.device, self.device)
		):
			msg = (
				"Render outputs are not on requested CUDA device. "
				f"rgb={rgb.device}, depth={depth.device}, requested={self.device}. "
				"This usually means a CPU fallback in the current PyTorch3D build/runtime."
			)
			if enforce_cuda_outputs:
				raise RuntimeError(msg)
			print(f"[WARN] {msg}")
		return rgb, depth


def _read_rgb_image(rgb_path: Path) -> np.ndarray:
	rgb_path = mesh_from_DSM._resolve_existing_path(rgb_path, Path(__file__).resolve().parent.parent)
	pil_image_module = importlib.import_module("PIL.Image")
	with pil_image_module.open(rgb_path) as img:
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
	rgb_u8 = (rgb.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
	pil_image_module = importlib.import_module("PIL.Image")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	pil_image_module.fromarray(rgb_u8, mode="RGB").save(out_path)


def _save_depth_image(depth: torch.Tensor, out_path: Union[str, Path]) -> None:
	out_path = Path(out_path)
	depth_np = depth.detach().cpu().numpy().astype(np.float32)
	pil_image_module = importlib.import_module("PIL.Image")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	pil_image_module.fromarray(depth_np, mode="F").save(out_path)


def _save_depth_preview_image(depth: torch.Tensor, out_path: Union[str, Path]) -> None:
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
	pil_image_module = importlib.import_module("PIL.Image")
	out_path.parent.mkdir(parents=True, exist_ok=True)
	pil_image_module.fromarray(preview, mode="L").save(out_path)


def _show_rendered(rgb: torch.Tensor, depth: torch.Tensor) -> None:
	plt = importlib.import_module("matplotlib.pyplot")
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
	mesh_ply_path: Optional[Union[str, Path]] = None,
	show_rendered_images: bool = True,
	nodata_value: Optional[float] = None,
	render_mode: str = "texture_only",
	render_faces_per_pixel: int = 1,
	render_blur_radius: float = 0.0,
	render_bin_size: Optional[int] = None,
	render_max_faces_per_bin: Optional[int] = None,
	enforce_cuda_outputs: bool = False,
) -> mesh_from_DSM:
	"""Run the DSM-to-render pipeline with fixed camera convention R_img2world+flip_xy."""
	mesh = mesh_from_DSM(dsm_tiff_path=dsm_tiff_path, vrt_path=vrt_path, nodata_value=nodata_value)
	xg, yg, zg = mesh.decode_xyz()

	valid_count = int(mesh.valid_mask.sum().item())
	total_count = int(mesh.valid_mask.numel())
	x_min, x_max = _finite_min_max(xg)
	y_min, y_max = _finite_min_max(yg)
	z_min, z_max = _finite_min_max(zg)
	print(f"Computation device: {mesh.device} | Xg device: {xg.device} | Zg device: {zg.device}")
	print(f"Initialized mesh_from_DSM: H={mesh.H}, W={mesh.W}")
	print(f"Valid pixels: {valid_count}/{total_count}")
	print(f"X range: [{x_min:.6f}, {x_max:.6f}] | Y range: [{y_min:.6f}, {y_max:.6f}] | Z range: [{z_min:.6f}, {z_max:.6f}]")

	rgb = _read_rgb_image(Path(rgb_path))
	if rgb.shape[:2] != (mesh.H, mesh.W):
		raise ValueError(f"RGB shape {rgb.shape[:2]} does not match DSM shape {(mesh.H, mesh.W)}")

	print(f"Loaded RGB overlay: shape={rgb.shape}, dtype={rgb.dtype}")
	rgb_cuda = torch.from_numpy(rgb).to(mesh.device)
	mesh.build_textured_mesh(rgb_cuda)
	print(f"Built textured mesh: verts={mesh.vertices.shape[0]}, faces={mesh.faces.shape[0]}")
	if mesh.render_origin_world is not None:
		print(f"Render origin (world): {mesh.render_origin_world.detach().cpu().numpy()}")

	if mesh_ply_path is not None:
		export_ply(mesh, str(mesh_ply_path))

	export_xyzrgb_points_to_ply(xg, yg, zg, rgb_cuda, path=str(points_ply_path))
	print(f"Exported colored point cloud to: {points_ply_path}")

	camera = mesh.create_perspective_camera_from_rt(
		R=rotation_matrix,
		T=translation_vector,
		focal_length_px=focal_length_px,
		principal_point_px=principal_point_px,
		image_size=image_size,
	)

	render_rgb, render_depth = mesh.render_perspective(
		camera=camera,
		image_size=image_size,
		faces_per_pixel=render_faces_per_pixel,
		blur_radius=render_blur_radius,
		bin_size=render_bin_size,
		max_faces_per_bin=render_max_faces_per_bin,
		render_mode=render_mode,
		enforce_cuda_outputs=enforce_cuda_outputs,
	)
	valid_depth = int(torch.isfinite(render_depth).sum().item())
	total_depth = int(render_depth.numel())
	print("Perspective camera parameters (fixed convention: R_img2world+flip_xy):")
	print(f"Render mode: {render_mode}")
	print(f"Focal length (px): {camera.focal_length[0].detach().cpu().numpy()}")
	print(f"Principal point (px): {camera.principal_point[0].detach().cpu().numpy()}")
	print(f"Image size (H, W): {image_size}")
	print(f"Rendered valid depth pixels: {valid_depth}/{total_depth}")
	print(f"Rendered RGB range: [{float(render_rgb.min().item()):.6f}, {float(render_rgb.max().item()):.6f}]")
	if valid_depth > 0:
		depth_valid = render_depth[torch.isfinite(render_depth)]
		print(f"Rendered depth range (valid): [{float(depth_valid.min().item()):.6f}, {float(depth_valid.max().item()):.6f}]")

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
	import numpy as np

	R = np.loadtxt("./ign_samples/dsm_ortho_2_img/R.txt", dtype=np.float32).reshape(3, 3)
	T = np.loadtxt("./ign_samples/dsm_ortho_2_img/T.txt", dtype=np.float32).reshape(3)

	main(
		dsm_tiff_path="./ign_samples/dsm_ortho_2_img/rge_dsm_corel.tif",
		vrt_path="./ign_samples/dsm_ortho_2_img/rge_dsm_corel.vrt",
		rgb_path="./ign_samples/dsm_ortho_2_img/rge_ortho.tif",
        # dsm_tiff_path="./data/chantier_lilles/camelia/mns/mns.0.tif",
        # vrt_path="./data/chantier_lilles/camelia/mns/mns_crop_camelia25cm.vrt",
        # rgb_path="./data/chantier_lilles/camelia/mns/ortho_25cm.tif",
        rotation_matrix=R,
		translation_vector=T,
		focal_length_px=1396.939,
		principal_point_px=(965.995, 539.776),
		image_size=(1080, 1920),
		points_ply_path="./ign_samples/dsm_ortho_2_img/output/points_rgb.ply",
		rendered_rgb_path="./ign_samples/dsm_ortho_2_img/output/rgb.png",
		rendered_depth_path="./ign_samples/dsm_ortho_2_img/output/depth.tif",
		mesh_ply_path="./ign_samples/dsm_ortho_2_img/output/mesh.ply",
		show_rendered_images=True,
		nodata_value=None,
	)
