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
    # Pw = torch.stack([Xg, Yg, Zg], dim=-1).reshape(-1, 3)

    # Pc = (R @ (Pw - C).T).T
    # u = K[0,0] * Pc[:,0] / Pc[:,2] + K[0,2]
    # v = K[1,1] * Pc[:,1] / Pc[:,2] + K[1,2]
    u = []
    v = []
    for y in range(image_h):
        for x in range(image_w):
            u.append(x / (image_w - 1))
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
    cx = (Xmin + Xmax) / 2
    cy = (Ymin + Ymax) / 2
    # Place camera below the mesh so mesh Z_cam = Z_world - camera_z > 0
    # (PyTorch3D cameras look in +Z in camera space with R=I)
    cz = Zmin -1.0  # Place camera below the mesh

    R = torch.eye(3, device=device).unsqueeze(0)
    T = torch.tensor([[-cx, -cy, -cz]], device=device, dtype=torch.float32)
    # T = torch.tensor([[0, 0, -cz]], device=device, dtype=torch.float32)
    # scale_x = 2.0 / (Xmax - Xmin)
    # scale_y = 2.0 / (Ymax - Ymin)

    # cx = 0.0
    # cy = 0.0
    
    scale_x = 0.02
    scale_y = 0.02


    return OrthographicCameras(
        device=device,
        R=R,
        T=T,
        focal_length=((scale_x, scale_y),),
        principal_point=((0,0),)
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

