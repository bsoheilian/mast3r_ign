"""Renderer accuracy diagnostics for OrientedImageTexturedDSMMesh.

camera_error_map compares, for every rendered pixel, the PyTorch3D camera with
STOrientation.LocalToImage: the surface point under the pixel is rebuilt from
the rasterizer barycentrics and reprojected in float64. A correct renderer gives
errors well below 0.01 px. This check isolated the half-pixel principal point
convention and the TF32 precision loss (~6 px on aerial cameras); see the
accuracy notes in mesh_from_dsm_oriented.
"""

from pathlib import Path

import numpy as np
import torch
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings

from dchan.core.utils.stereopolis_orientation import STOrientation


def camera_error_map(mesh, ori: STOrientation, out_png: Path, label: str) -> None:
	"""Print per-pixel camera error statistics and save an error map, for a built mesh."""
	import matplotlib.pyplot as plt

	w, h = ori.intrinsic.image_size
	cam = mesh.create_perspective_camera_from_orientation(ori)
	frag = MeshRasterizer(cameras=cam, raster_settings=RasterizationSettings(image_size=(h, w)))(mesh.mesh)
	p2f = frag.pix_to_face[0, ..., 0]
	hit = p2f >= 0
	print(f"[{label}] hits={hit.sum().item()}")
	if not hit.any():
		return
	vv, uu = torch.nonzero(hit, as_tuple=True)
	b = frag.bary_coords[0, ..., 0, :][hit].double().unsqueeze(-1)
	X = (b * mesh.vertices[mesh.faces[p2f[hit]]].double()).sum(1) + mesh.render_origin_world.double()
	X = X.cpu().numpy()

	R = ori.extrinsic.get_rot(image2ground=False)
	C = ori.extrinsic.get_center()
	p = (X - C) @ R.T
	f, cx, cy = ori.intrinsic.ppa["f"], ori.intrinsic.ppa["cx"], ori.intrinsic.ppa["cy"]
	proj = np.stack([f * p[:, 0] / p[:, 2] + cx, f * p[:, 1] / p[:, 2] + cy], 1)
	pix = np.stack([uu.cpu().numpy(), vv.cpu().numpy()], 1).astype(float)
	err = proj - pix

	# err ~ offset + B @ (u - cx, v - cy); unreliable when the principal point lies far outside the image
	A = np.column_stack([np.ones(len(pix)), pix[:, 0] - cx, pix[:, 1] - cy])
	coef, *_ = np.linalg.lstsq(A, err, rcond=None)
	print(f"[{label}] size={w}x{h} f={f:.3f} cx={cx:.3f} cy={cy:.3f} C={C} det(R)={np.linalg.det(R):.6f} "
		  f"orth={np.abs(R @ R.T - np.eye(3)).max():.2e} depth~{np.median(p[:, 2]):.1f}")
	print(f"[{label}] err mean={err.mean(0)} max={np.abs(err).max(0)}")
	print(f"[{label}] fit offset={coef[0]}  linear=\n{coef[1:].T}")

	m = np.full((h, w), np.nan)
	m[pix[:, 1].astype(int), pix[:, 0].astype(int)] = np.linalg.norm(err, axis=1)
	plt.figure(figsize=(7, 5))
	plt.imshow(m)
	plt.colorbar()
	plt.title(f"{label} camera error (px)")
	plt.savefig(out_png)
	plt.close()
