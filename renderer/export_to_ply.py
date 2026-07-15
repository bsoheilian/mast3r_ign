import numpy as np
import open3d as o3d

def export_ply(mesh, path="debug.ply"):


    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(faces)

    m.compute_vertex_normals()

    o3d.io.write_triangle_mesh(path, m)
    print("Saved:", path)

def export_xyzrgb_points_to_ply(Xg, Yg, Zg, rgb, path="points_rgb.ply"):


    # --- tensors → numpy ---
    if hasattr(Xg, "detach"):
        Xg = Xg.detach().cpu().numpy()
        Yg = Yg.detach().cpu().numpy()
        Zg = Zg.detach().cpu().numpy()

    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()

    # --- ensure correct dtype ---
    rgb = rgb[..., :3]

    # 🔥 CRITICAL FIX
    if rgb.max() <= 1.0:
        rgb = (rgb * 255.0)

    rgb = rgb.astype(np.uint8)

    pts = np.stack([Xg, Yg, Zg], axis=-1).reshape(-1, 3)
    col = rgb.reshape(-1, 3)
    print(f"Points shape: {pts.shape}, dtype: {pts.dtype}, min: {pts.min()}, max: {pts.max()}")
    print(f"Colors shape: {col.shape}, dtype: {col.dtype}, min: {col.min()}, max: {col.max()}")

    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    col = col[mask]

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(pts, col):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

    print("Saved:", path)

