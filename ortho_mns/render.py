# debug_render.py
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex
)

def create_debug_renderer(image_size=800, device="cuda"):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    cameras = PerspectiveCameras(
        device=device,
        in_ndc=True,   # default, fine for debugging
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
        ),
    )
    return renderer, cameras


def export_textured_obj_fixed(
    verts,
    faces,
    verts_uvs,
    texture_image,
    obj_path="mesh.obj",
    texture_name="texture.png"
):
    import numpy as np
    from imageio import imwrite

    verts = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    uv = verts_uvs.detach().cpu().numpy()

    # OBJ UV convention fix
    uv[:, 1] = 1.0 - uv[:, 1]

    # Save texture
    imwrite(texture_name, texture_image.astype(np.uint8))

    mtl_path = obj_path.replace(".obj", ".mtl")

    # --- MTL ---
    with open(mtl_path, "w") as f:
        f.write(f"""newmtl mat0
Ka 1 1 1
Kd 1 1 1
Ks 0 0 0
d 1
illum 2
map_Kd {texture_name}
""")

    # --- OBJ ---
    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_path.split('/')[-1]}\n")
        f.write("usemtl mat0\n")

        # vertices
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # UVs (IMPORTANT: independent indexing)
        for t in uv:
            f.write(f"vt {t[0]} {t[1]}\n")

        # faces
        for face in faces:
            # OBJ uses 1-based indexing
            v1, v2, v3 = face + 1

            # IMPORTANT: same index used for vt ONLY works
            # if vertex order == uv order (your case TRUE)
            f.write(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n")

def export_ply(mesh, path="debug.ply"):
    import open3d as o3d

    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(faces)

    m.compute_vertex_normals()

    o3d.io.write_triangle_mesh(path, m)
    print("Saved:", path)

def export_xyzrgb_points_to_ply(Xg, Yg, Zg, rgb, path="points_rgb.ply"):
    import numpy as np

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

# ----------------------------
# Minimal WORKING example
# ----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simple square (2 triangles)
    verts = torch.tensor(
        [[-1, -1, 0],
         [ 1, -1, 0],
         [ 1,  1, 0],
         [-1,  1, 0]],
        dtype=torch.float32,
        device=device,
    )

    faces = torch.tensor(
        [[0, 1, 2],
         [0, 2, 3]],
        dtype=torch.int64,
        device=device,
    )

    # Vertex colors (RGB) — THIS IS THE KEY FIX
    verts_rgb = torch.tensor(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 1, 0]],
        dtype=torch.float32,
        device=device,
    )

    textures = TexturesVertex(verts_features=verts_rgb[None])

    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures,
    )

    renderer, cameras = create_debug_renderer(image_size=512, device=device)

    image = renderer(mesh)[0, ..., :3]  # (H, W, 3)

    # Save for inspection
    from PIL import Image
    Image.fromarray((image.cpu().numpy() * 255).astype("uint8")).save("debug_render.png")

    print("Saved debug_render.png")