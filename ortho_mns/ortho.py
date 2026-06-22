import math

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.renderer import OrthographicCameras

def build_dsm_mesh(Xg, Yg, Zg):
    H, W = Zg.shape
    device = Zg.device

    verts = torch.stack([Xg, Yg, Zg], dim=-1).reshape(-1, 3)

    faces = []
    for y in range(H - 1):
        for x in range(W - 1):
            v0 = y * W + x
            v1 = v0 + 1
            v2 = v0 + W
            v3 = v2 + 1
            faces += [[v0, v1, v2], [v1, v3, v2]]

    faces = torch.tensor(faces, device=device)
    print(f"Mesh vertices shape: {verts.shape}, faces shape: {faces.shape}")
    return verts, faces

def build_uvs_from_projection(Xg, Yg, Zg, R, C, K, image_h, image_w):
    Pw = torch.stack([Xg, Yg, Zg], dim=-1).reshape(-1, 3)

    Pc = (R @ (Pw - C).T).T
    u = K[0,0] * Pc[:,0] / Pc[:,2] + K[0,2]
    v = K[1,1] * Pc[:,1] / Pc[:,2] + K[1,2]

    u = u / (image_w - 1)
    v = 1.0 - v / (image_h - 1)

    print(f"UVs shape: {u.shape}, {v.shape}")

    verts_uvs = torch.stack([u, v], dim=-1)
    print(f"verts_uvs shape: {verts_uvs.shape}")

    return verts_uvs

def __build_uvs_from_projection(Xg, Yg, Zg, R, C, K, image_h, image_w):
    Pw = torch.stack([Xg, Yg, Zg], dim=-1).reshape(-1, 3)

    # Pc = (R.T @ (Pw - C).T).T
    R_inv = R.T
    Pc = (R_inv @ (Pw - C).reshape(-1, 3).T).T.reshape(-1, 3)
    uu = K[0,0] * Pc[:,0] / Pc[:,2] + K[0,2]
    vv = K[1,1] * Pc[:,1] / Pc[:,2] + K[1,2]
    uu = uu / (image_w - 1)
    vv = 1.0 - vv / (image_h - 1)


    u = []
    v = []
    for y in range(image_h):
        for x in range(image_w):
            u.append(1.0 * x / (image_w - 1))
            v.append(1.0 - y / (image_h - 1))
    # print(f"UVs shape: {len(u)}, {len(v)} (expected: {image_h * image_w})")
    u = torch.tensor(u, device=Xg.device)
    v = torch.tensor(v, device=Xg.device)
    # u = u / (image_w - 1)
    # v = 1.0 - v / (image_h - 1)

    verts_uvs = torch.stack([u, v], dim=-1)
    # print(f"__verts_uvs shape: {verts_uvs.shape}")
    # print(f"mesh vertex count expectation: {Xg.numel()}")
    # print(f"Pw: {Pw.shape[0]}")
    
    max_dif_u = (uu-u).abs().max()
    max_dif_v = (vv-v).abs().max()
    print(f"Max difference between computed and expected u: {max_dif_u.item()}")
    print(f"Max difference between computed and expected v: {max_dif_v.item()}")


    return verts_uvs


def build_textured_mesh(verts, faces, verts_uvs, image):
    # TexturesUV expects maps in (N, H, W, C) format
    if image.shape[0] in (1, 3, 4) and image.ndim == 3:
        image = image.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    textures = TexturesUV(
        maps=image.unsqueeze(0),  # (1, H, W, C)
        verts_uvs=verts_uvs.unsqueeze(0),
        faces_uvs=faces.unsqueeze(0)
    )

    return Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )

def create_topdown_ortho_camera(Xmin, Xmax, Ymin, Ymax, Zmin, device):
    # Place camera below the mesh so mesh Z_cam = Z_world - camera_z > 0
    # (PyTorch3D cameras look in +Z in camera space with R=I)
    default_param = False
    if default_param:
        # Use default parameters for orthographic camera
        cx = (Xmin + Xmax) / 2
        cy = (Ymin + Ymax) / 2
        cz = Zmin - 1  # Place camera below the mesh
        R = torch.eye(3, device=device).unsqueeze(0)
        print(f"Camera rotation:\n{R} {type(R)}, shape: {R.shape}")
        T = torch.tensor([[-cx, -cy, -cz]], device=device, dtype=torch.float32)
        scale_x = 0.03
        scale_y = 0.03
        px = 0.5
        py = 0.5
    else : 
        # Use custom parameters for orthographic camera
        # cx = (Xmin + Xmax) / 2
        # cy = (Ymin + Ymax) / 2
        # cz = Zmin + 20.  # Place camera below the mesh
        # print(f"Camera position: ({cx:.2f}, {cy:.2f}, {cz:.2f})")
        # cx = -2.11
        # cy = 0.13
        # cz = 1.0
        
        # this is the coordinate on the ground plane that the camera is looking at (the center of the ortho rectification)
        cx = -3.0
        cy = 5.0
        cz = 1.0
        # the above center together with R = torch.tensor([[-1,  0,  0],[0, 1,  0],[0,  0, 1]], device=device, dtype=torch.float32)
        #lead to a good looking ortho


        R_x = torch.tensor([[1,  0,  0],[0, -1,  0],[0,  0, -1]], device=device, dtype=torch.float32)
        R_z = torch.tensor([[-1, 0, 0],[0,-1, 0],[0, 0, 1]],device=device, dtype=torch.float32)
        R = (R_z @ R_x)
        

        print(f"Camera rotation:\n{R} {type(R)}, shape: {R.shape}")
        C = torch.tensor([cx, cy, cz], device=device, dtype=torch.float32)
        # T = -C.unsqueeze(0) 
        T = -1*torch.matmul(R, 1.0*C)
        # T = C
        # T = T.unsqueeze(0)
        R = R.unsqueeze(0)  # Add batch dimension
        T = T.unsqueeze(0)  # Add batch dimension
        gsd = 0.1
        scale_x = 2.0/(Xmax - Xmin) 
        scale_y = 2.0/(Ymax - Ymin)
        px = 0.0
        py = 0.0
        gsd = 0.1  # Ground Sample Distance in world units per pixel    
        W = math.ceil((Xmax - Xmin) / gsd)
        H = math.ceil((Ymax - Ymin) / gsd)

    return OrthographicCameras(
        device=device,
        R=R,
        T=T,
        focal_length=((scale_x, scale_y),),
        principal_point=((px,py),),
        image_size=((H, W),)

    )

from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer,
    RasterizationSettings, SoftPhongShader, AmbientLights
)

def create_ortho_renderer(image_size, camera):
    raster = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    device = camera.device
    # Full ambient light so texture colors are rendered as-is without shading
    lights = AmbientLights(device=device)

    return MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster
        ),
        shader=SoftPhongShader(device=device, cameras=camera, lights=lights)
    )

def render_ortho(mesh, renderer):
    return renderer(mesh)[0, ..., :3]

def draw_textured_mesh_open3d(
    mesh_p3d,
    verts_uvs,
    texture_image,
    show_back_faces=True,
):
    """
    Visualize a textured PyTorch3D mesh in Open3D.

    Args:
        mesh_p3d: PyTorch3D Meshes object (batch size = 1)
        verts_uvs: (V, 2) torch tensor in [0,1], same vertex order
        texture_image: (H, W, 3) uint8 numpy array
        show_back_faces: show both sides (useful for terrain)
    """
    import open3d as o3d
    import numpy as np
    import torch

    # --- Extract geometry ---
    verts = mesh_p3d.verts_packed().detach().cpu().numpy()
    faces = mesh_p3d.faces_packed().detach().cpu().numpy()
    verts_uvs = verts_uvs.detach().cpu().numpy()

    assert verts.shape[0] == verts_uvs.shape[0], "UV/vertex count mismatch"
    assert texture_image.dtype == np.uint8, "Texture must be uint8"

    # --- Open3D mesh ---
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

    # --- Open3D requires per-triangle UVs ---
    triangle_uvs = verts_uvs[faces].reshape(-1, 2)
    mesh_o3d.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)

    # --- Open3D uses bottom-left UV origin ---
    texture_image = np.ascontiguousarray(np.flipud(texture_image))
    mesh_o3d.textures = [o3d.geometry.Image(texture_image)]

    # --- Disable vertex colors (they override textures) ---
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector()

    # --- Normals ---
    mesh_o3d.compute_vertex_normals()

    # --- Visualize ---
    o3d.visualization.draw_geometries(
        [mesh_o3d],
        mesh_show_back_face=show_back_faces
    )