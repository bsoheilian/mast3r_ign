from render import create_debug_renderer
import torch
import numpy as np
from ortho import build_dsm_mesh, build_uvs_from_projection, build_textured_mesh, create_topdown_ortho_camera, create_ortho_renderer, render_ortho
from ortho import __build_uvs_from_projection
try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        OrthographicCameras,
        RasterizationSettings,
        MeshRasterizer,
        TexturesVertex,
    )
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False

def depth_to_3d(depth, K, R, t, device='cuda'):
    """
    Convert depth map to 3D points in world coordinates.
    
    Args:
        depth: (H, W) tensor, per-pixel depth in camera coordinates.
        K: (3, 3) tensor or array, camera intrinsics matrix.
        R: (3, 3) tensor, rotation matrix (camera to world).
        t: (3,) tensor, translation vector (camera to world).
        device: str, device to use for computation.
    
    Returns:
        points_world: (H, W, 3) tensor, 3D points in world coordinates.
    """
    H, W = depth.shape
    u = torch.arange(W, device=device).float()
    v = torch.arange(H, device=device).float()
    u, v = torch.meshgrid(u, v, indexing='xy')
        # Convert K to torch tensor on device if needed
    if not isinstance(K, torch.Tensor):
        K = torch.from_numpy(K).float().to(device)
    else:
        K = K.to(device).float()
    
    
    # Backproject to camera coordinates
    Z = depth.to(device).float()
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    points_cam = torch.stack([X, Y, Z], dim=-1)  # (H, W, 3)
    
    # Transform to world coordinates (R is camera-to-world, t is camera-to-world translation)
    t = t.to(device).float() if isinstance(t, torch.Tensor) else torch.tensor(t, device=device, dtype=torch.float32)
    points_world = torch.einsum('ij,hwj->hwi', R, points_cam) + t
    return points_world

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rgb = np.load("./output/debug_rgb_Lille-150127_0485-11-00002_0000384.jpg_2.npy")
    
    rgb = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device)  # Convert to tensor and normalize 
    if rgb.max() > 1.0:
        rgb = rgb / 255.0  # Normalize to [0, 1] if needed
    print(f"RGB image shape: {rgb.shape}, dtype: {rgb.dtype}")

    
    depth = np.load("./output/debug_depthmap_Lille-150127_0485-11-00002_0000384.jpg_2.npy").reshape(-1, rgb.shape[1])
    print(f"Depth image shape: {depth.shape}, dtype: {depth.dtype}")
    depth = torch.from_numpy(depth).to(device)


    with open('/mast3r_ign/output/intrinsics.txt') as f:
        K = np.array(eval(f.read()))
        print(f"Camera intrinsics:\n{K} {type(K)}")

    with open('/mast3r_ign/output/rot_copy.txt') as f:
        R = np.array(eval(f.read()))
        print(f"Camera rotation:\n{R} {type(R)}")
        R = torch.from_numpy(R).float().to(device)

    with open('/mast3r_ign/output/trans.txt') as f:
        t = np.array(eval(f.read()))
        print(f"Camera translation:\\n{t} {type(t)}")
        t = torch.from_numpy(t).float().to(device)


    points_w = depth_to_3d(depth, K, R, t, device=device)
    print(f"3D points shape: {points_w.shape}, dtype: {points_w.dtype}")
    Xg, Yg, DSM = points_w[..., 0], points_w[..., 1], points_w[..., 2]

    verts, faces = build_dsm_mesh(Xg, Yg, DSM)
    # verts_uvs = build_uvs_from_projection(Xg, Yg, DSM, 
    #                                       R, t, K, depth.shape[0], depth.shape[1])
    
    verts_uvs = __build_uvs_from_projection(Xg, Yg, DSM, 
                                          R, t, K, depth.shape[0], depth.shape[1])

    mesh = build_textured_mesh(verts, faces, verts_uvs, rgb)

   


    # visualize_mesh_debug(mesh)

    Xmin, Xmax = Xg.min().item(), Xg.max().item()
    Ymin, Ymax = Yg.min().item(), Yg.max().item()
    print(f"Mesh bounds: X=[{Xmin:.2f}, {Xmax:.2f}], Y=[{Ymin:.2f}, {Ymax:.2f}], Z=[{DSM.min().item():.2f}, {DSM.max().item():.2f}]")
    # camera = create_topdown_ortho_camera(Xmin, Xmax, Ymin, Ymax, DSM.min(), device="cuda")
    camera = create_topdown_ortho_camera(Xmin, Xmax, Ymin, Ymax, DSM.min(), device="cuda")
    # to run this 

    renderer = create_ortho_renderer(image_size=(500, 500), camera=camera)

    ortho = render_ortho(mesh, renderer)
    print(f"Ortho image shape: {ortho.shape}, dtype: {ortho.dtype}")

    import matplotlib.pyplot as plt
    plt.imshow(ortho.cpu().numpy())
    plt.show()


if __name__ == "__main__":
    main()