import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.renderer import OrthographicCameras
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer,
    RasterizationSettings, SoftPhongShader
)
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

class OrthoRectifier:
    def __init__(self, rgb_img: np.ndarray, depth_img: np.ndarray, intrinsics: np.ndarray, 
                 rotation: np.ndarray, translation: np.ndarray) -> None:
        self.rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float() #/255.0  # Convert to tensor and add batch dimension
        self.rgb_img = self.rgb_img.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        assert self.rgb_img.is_cuda

        self.depth_img = torch.from_numpy(depth_img).unsqueeze(0).float()  # Convert to tensor and add batch and channel dimensions
        self.depth_img = self.depth_img.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        assert self.depth_img.is_cuda 

        self.intrinsics = intrinsics
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)
        self.rotation = rotation
        self.translation = translation

    def _creat_point_cloud(self):
        # Create a grid of pixel coordinates
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]
        f = self.intrinsics[0, 0]  # Assuming fx = fy

        H, W = self.depth_img.shape[1:3]
        print(f'H: {H}, W: {W}')
        device = self.depth_img.device
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij")
        
        z = self.depth_img  # Remove batch dimension
        x = (xs - cx) * z / f
        y = (ys - cy) * z / f
        points_cam = torch.stack([x, y, z], dim=-1)      # [H,W,3]
        points_cam = points_cam.reshape(-1, 3)

        # Apply extrinsics (rotation and translation)
        rotation_torch = torch.from_numpy(self.rotation).float().to(self.depth_img.device)
        translation_torch = torch.from_numpy(self.translation).float().to(self.depth_img.device)
        points_world = (rotation_torch @ points_cam.T).T + translation_torch
        
        return points_world
    
