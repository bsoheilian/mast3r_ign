import numpy as np
import torch
import torch.nn.functional as F

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

    def _visualize_point_cloud(self, points_cam: torch.Tensor):
        """Display point cloud in 3D using GPU acceleration with Open3D"""
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D not installed. Install it with: pip install open3d")
        
        # Convert to numpy (stays on GPU computation, transferred only for visualization)
        points_np = points_cam.cpu().numpy()
        
        # Extract colors from normalized RGB image (already normalized to [0,1])
        H, W = self.depth_img.shape[1:3]
        rgb_colors = self.rgb_img.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
        rgb_colors = rgb_colors.cpu().numpy()
        colors = rgb_colors.reshape(-1, 3)  # Flatten to [H*W, 3]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize with Open3D (GPU-accelerated)
        o3d.visualization.draw_geometries([pcd], window_name='3D Point Cloud (GPU Accelerated)')

    def run(self):
        points_cam = self._creat_point_cloud()
        self._visualize_point_cloud(points_cam)
        

