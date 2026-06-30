
import sys
from render import export_ply, export_xyzrgb_points_to_ply

import torch
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ori_img_utils.ori import Orientation
from renderer.mesh_from_3D import TexturedMesh3D
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k_path = '/mast3r_ign/output/K.txt'
    R_path = '/mast3r_ign/output/R.txt'
    C_path = '/mast3r_ign/output/T.txt'
    rgb_path = '/mast3r_ign/output/rgb.npy'
    depth_path = '/mast3r_ign/output/depth.npy'
    
    # Load orientation from files 
    ori = Orientation.from_files(k_path, R_path, C_path)
    print(ori)

    # Load RGB and depth images
    rgb = np.load(rgb_path)  # Load RGB image With shape (H, W, 3)
    H, W, _ = rgb.shape
    print(f"RGB image shape: {rgb.shape}, dtype: {rgb.dtype}, min: {rgb.min()}, max: {rgb.max()}, type: {type(rgb)}")
    # plt.imshow(rgb)
    # plt.show()
    depth = np.load(depth_path)  # Load depth image With shape (HxW,)
    depth = depth.reshape(H, -1)  # Reshape to (H, W)
    print(f"Depth image shape: {depth.shape}, dtype: {depth.dtype}, type: {type(depth)}, min: {depth.min()}, max: {depth.max()}")
    # plt.imshow(depth, cmap='gray', vmin=2.0, vmax=2.50)
    # plt.show()

    #convert images to torch tensors
    # rgb = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) # Convert to torch tensor of shape (3, H, W)
    # depth = torch.from_numpy(depth).float().to(device)  

    # apply K
    Xcam, Ycam, Zcam = ori.apply_K_torch(depth, z_scale=2.5, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Xcam shape: {Xcam.shape}, Ycam shape: {Ycam.shape}, Zcam shape: {Zcam.shape}")
    print(f"Xcam type: {Xcam.dtype}, Ycam type: {Ycam.dtype}, Zcam type: {Zcam.dtype}")

    rgb_copy = np.load(rgb_path) 
    export_xyzrgb_points_to_ply(Xcam, Ycam, Zcam, 255*rgb_copy, path="./output/points_cam_refactor.ply")

    # apply R and t
    Xg, Yg, Zg = ori.apply_ext(Xcam, Ycam, Zcam, direction="cam2world")
    print(f"Xg shape: {Xg.shape}, Yg shape: {Yg.shape}, Zg shape: {Zg.shape}")  
    export_xyzrgb_points_to_ply(Xg, Yg, Zg, 255*rgb_copy, path="./output/points_world_refactor.ply")
    
    mesh = TexturedMesh3D(Xg, Yg, Zg, rgb, triangle_filter = True, triangle_max_edge_length = 0.5)
    export_ply(mesh.mesh, path="./output/refactor_mesh_filtered.ply") 

    ortho = mesh.create_orth(
        gsd=0.05,
        show=False,
        profile=True,
        bin_size=None,       # None = PyTorch3D picks coarse bin size automatically
        max_faces_per_bin=None,  # None = safe_raster auto-computes a safe value
        safe_raster=True
    )

    # Save ortho image to file
    ortho_np = (ortho.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    out_path = "./output/refactor_ortho_filtered.png"
    Image.fromarray(ortho_np).save(out_path)
    print(f"Ortho saved to {out_path}  shape={ortho_np.shape}")

    


if __name__ == "__main__":
    main()