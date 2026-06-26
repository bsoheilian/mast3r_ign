from render import create_debug_renderer, export_ply, export_textured_obj_fixed, export_xyzrgb_points_to_ply
import torch
import numpy as np
from ortho import build_dsm_mesh, build_uvs_from_projection, build_textured_mesh, create_topdown_ortho_camera, create_ortho_renderer, draw_textured_mesh_open3d, render_ortho, write_rgb_vrt, write_rgb_vrt_qgis_safe
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

def depth_to_3d(depth, K, R, t, device='cuda', scale_factor=1.0):
    """
    Convert depth map to 3D points in world coordinates.
    
    Args:
        depth: (H, W) tensor, per-pixel depth in camera coordinates.
        K: (3, 3) tensor or array, camera intrinsics matrix.
        R: (3, 3) tensor, rotation matrix (camera to world).
        t: (3,) tensor, translation vector (camera to world).
        device: str, device to use for computation.
        scale_factor: float, factor to scale the depth values.
    
    Returns:
        points_world: (H, W, 3) tensor, 3D points in world coordinates.
    """
    H, W = depth.shape
    print(f"depth shape: {depth.shape}, dtype: {depth.dtype}")
    u = torch.arange(W, device=device, dtype=depth.dtype)
    v = torch.arange(H, device=device, dtype=depth.dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    
    # u, v = torch.meshgrid(v, u, indexing='ij')  # <-- IMPORTANT FIX
    
    # Convert K to torch tensor on device if needed
    if not isinstance(K, torch.Tensor):
        K = torch.from_numpy(K).to(device).to(dtype=depth.dtype)
    else:
        K = K.to(device).to(dtype=depth.dtype)
    
    # Backproject to camera coordinates
    Z = depth.to(device).to(dtype=depth.dtype) * scale_factor   
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    points_cam = torch.stack([X, Y, Z], dim=-1)  # (H, W, 3)
    
    # Transform to world coordinates (R is camera-to-world, t is camera-to-world translation)
    t = t.to(device).to(dtype=depth.dtype) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device, dtype=depth.dtype)
    points_world = torch.einsum('ij,hwj->hwi', R, points_cam) + t
    
    PW = R @ points_cam.reshape(-1, 3).T + t.view(3, 1)
    PW = PW.T.reshape(H, W, 3)
    print(f"points_world shape: {points_world.shape}, dtype: {points_world.dtype}")
    print(f"PW shape: {PW.shape}, dtype: {PW.dtype}")
    print(f"Difference between points_world and PW: {(points_world - PW).abs().max()}")

    # Apply inverse R and T to retrieve points_cam
    R_inv = R.T
    points_cam_retrieved = (R_inv @ (PW - t).reshape(-1, 3).T).T.reshape(H, W, 3)
    
    print(f"points_cam_retrieved shape: {points_cam_retrieved.shape}, dtype: {points_cam_retrieved.dtype}")
    print(f"Difference between original points_cam and retrieved: {(points_cam - points_cam_retrieved).abs().max()}")
    
    return points_world

def main():
    import matplotlib.pyplot as plt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rgb_int = np.load("./output/debug_rgb_Lille-150127_0485-11-00002_0000384.jpg_2.npy")
    # plt.imshow(rgb_int)
    # plt.show()

    print(f"first read rgb : {rgb_int.min()}, {rgb_int.max()}, {rgb_int.dtype}")
    rgb = torch.from_numpy(rgb_int).permute(2, 0, 1).float().to(device)  # Convert to tensor and normalize 
    if rgb.max() > 1.0:
        rgb = rgb / 255.0  # Normalize to [0, 1] if needed
    print(f"RGB image shape: {rgb.shape}, dtype: {rgb.dtype}")

    
    depth = np.load("./output/debug_depthmap_Lille-150127_0485-11-00002_0000384.jpg_2.npy").reshape(-1, rgb.shape[2])
    print(f"Depth image shape: {depth.shape}, dtype: {depth.dtype}")
    
    # plt.imshow(depth)
    # plt.show()
    
    depth = torch.from_numpy(depth).to(device)

 

    with open('/mast3r_ign/output/intrinsics.txt') as f:
        K = np.array(eval(f.read()))
        print(f"Camera intrinsics:\n{K} {type(K)}")

    with open('/mast3r_ign/output/rot_copy.txt') as f:
        R = np.array(eval(f.read()))
        print(f"Camera rotation:\n{R} {type(R)}")
        # R = torch.from_numpy(R).double().to(device)  # Use float64 for better precision
        R = torch.from_numpy(R).float().to(device)  # Use float64 for better precision
        
        # # Check if R is orthogonal
        # R_RTR = R @ R.T
        # print(f"R @ R.T (should be identity):\n{R_RTR}")
        # print(f"Max deviation from identity: {(R_RTR - torch.eye(3, device=device, dtype=torch.float64)).abs().max()}")

    with open('/mast3r_ign/output/trans_copy.txt') as f:
        t = np.array(eval(f.read()))
        print(f"Camera translation:\\n{t} {type(t)}")
        # t = torch.from_numpy(t).double().to(device)  # Use float64
        t = torch.from_numpy(t).float().to(device)  # Use float64

    # Convert depth and K to float64 for better precision
    # depth = depth.double()
    if not isinstance(K, torch.Tensor):
        # K = torch.from_numpy(K).double().to(device)
        K = torch.from_numpy(K).float().to(device)
    else:
        # K = K.double().to(device)
        K = K.float().to(device)

    points_w = depth_to_3d(depth, K, R, t, device=device, scale_factor=2.5)
    print(f"3D points shape: {points_w.shape}, dtype: {points_w.dtype}")
    Xg, Yg, DSM = points_w[..., 0], points_w[..., 1], points_w[..., 2]

     

    verts, faces = build_dsm_mesh(Xg, Yg, DSM)
    # verts_uvs = build_uvs_from_projection(Xg, Yg, DSM, 
    #                                       R, t, K, depth.shape[0], depth.shape[1])
    
    verts_uvs = __build_uvs_from_projection(Xg, Yg, DSM, 
                                          R, t, K, depth.shape[0], depth.shape[1])

     
    mesh = build_textured_mesh(verts, faces, verts_uvs, rgb)
    rgb_input = rgb_int[:, :, :3].astype("uint8")
    # export_textured_obj_fixed(
    #     verts=mesh.verts_packed(),
    #     faces=mesh.faces_packed(),
    #     verts_uvs=verts_uvs,
    #     texture_image=rgb_input,
    #     obj_path="./output/debug_mesh.obj",
    #     texture_name="./output/texture.png"
    #     )
    export_ply(mesh, path="./output/debug_mesh_2_Trans.ply")    
    # draw_textured_mesh_open3d(mesh_p3d=mesh,verts_uvs=verts_uvs,texture_image=rgb_input,)
    
    export_xyzrgb_points_to_ply(Xg, Yg, DSM, 255*rgb_int, path="./output/points_2.ply")
    print(rgb_input.min(), rgb_input.max(), rgb_input.dtype)
    


    # visualize_mesh_debug(mesh)

    Xmin, Xmax = Xg.min().item(), Xg.max().item()
    Ymin, Ymax = Yg.min().item(), Yg.max().item()
    print(f"Mesh bounds: X=[{Xmin:.2f}, {Xmax:.2f}], Y=[{Ymin:.2f}, {Ymax:.2f}], Z=[{DSM.min().item():.2f}, {DSM.max().item():.2f}]")
    # camera = create_topdown_ortho_camera(Xmin, Xmax, Ymin, Ymax, DSM.min(), device="cuda")
    camera = create_topdown_ortho_camera(Xmin, Xmax, Ymin, Ymax, DSM.min(), device="cuda")
    # to run this 

    gsd = 0.1
    dx= Xmax - Xmin
    dy = Ymax - Ymin
    res_x = int(np.ceil(dx / gsd))
    res_y = int(np.ceil(dy / gsd))
    print(f"Orthographic image resolution: {res_x} x {res_y}")
    renderer = create_ortho_renderer(image_size=(res_y, res_x), camera=camera)

    ortho = render_ortho(mesh, renderer)

    print(f"Ortho image shape: {ortho.shape}, dtype: {ortho.dtype}")

    import matplotlib.pyplot as plt
    plt.imshow(ortho.cpu().numpy())
    plt.show()
    
    # Save ortho to PNG file
    from PIL import Image
    ortho_np = ortho.cpu().numpy()
    if ortho_np.ndim == 3 and ortho_np.shape[0] == 3:
        ortho_np = (ortho_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    elif ortho_np.max() <= 1.0:
        ortho_np = (ortho_np * 255).astype(np.uint8)
    ortho_path = "./output/ortho.png"
    Image.fromarray(ortho_np).save(ortho_path)

    with open('/mast3r_ign/output/trans_copy.txt') as f:
        _translation = np.array(eval(f.read()))
        print(f"Camera translation:\\n{_translation} {type(_translation)}")
        # t = torch.from_numpy(t).double().to(device)  # Use float64
        #_translation = torch.from_numpy(_translation).float().to(device)  # Use float64
    ortho_path = "/home/BSoheilian/work/dev/mast3r_0ign/output/ortho.png"
    # write_rgb_vrt("./output/ortho.vrt", ortho_path, res_x,res_y, Xmin+_translation[0], Ymax+_translation[1], gsd, gsd, epsg="EPSG:2154")
    write_rgb_vrt_qgis_safe("./output/ortho.vrt", ortho_path, res_x,res_y, Xmin+_translation[0], Ymin+_translation[1], gsd, epsg="EPSG:2154")



if __name__ == "__main__":
    main()