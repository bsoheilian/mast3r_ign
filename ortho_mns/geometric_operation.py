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
    
    def _build_uvs(self, H, W, device="cuda"):
        """
        Returns:
            verts_uvs: (V,2)
            faces_uvs: (F,3)
        """
        u = torch.linspace(0, 1, W, device=device)
        v = torch.linspace(1, 0, H, device=device)  # flip Y for images
        U, V = torch.meshgrid(u, v, indexing="xy")

        verts_uvs = torch.stack([U, V], dim=-1).reshape(-1, 2)

        faces_uvs = []
        for y in range(H - 1):
            for x in range(W - 1):
                v0 = y * W + x
                v1 = v0 + 1
                v2 = v0 + W
                v3 = v2 + 1
                faces_uvs.append([v0, v1, v2])
                faces_uvs.append([v1, v3, v2])

        faces_uvs = torch.tensor(faces_uvs, device=device)
        return verts_uvs, faces_uvs
    
    def _build_dsm_mesh(self, Xg, Yg, Zg):
        """
        Xg, Yg, Zg: (H,W) tensors
        Returns PyTorch3D Meshes object
        """
        H, W = Zg.shape
        device = Zg.device

        # vertices
        verts = torch.stack([Xg, Yg, Zg], dim=-1).reshape(-1, 3)

        # faces
        faces = []
        for y in range(H - 1):
            for x in range(W - 1):
                v0 = y * W + x
                v1 = v0 + 1
                v2 = v0 + W
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])

        faces = torch.tensor(faces, device=device)

        return verts, faces
    
    def _attach_texture(self, verts, faces, verts_uvs, faces_uvs, ortho_image):
        """
        ortho_image: (3,H,W), float in [0,1]
        """
        textures = TexturesUV(
            maps=ortho_image.unsqueeze(0),   # (1,3,H,W)
            faces_uvs=faces_uvs.unsqueeze(0),
            verts_uvs=verts_uvs.unsqueeze(0),
        )

        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )
        return mesh
    
    def create_ortho_camera(self,
        center,      # (3,) look-at point
        scale,       # controls zoom
        R=None,
        T=None,
        device="cuda"):

        if R is None:
            R = torch.eye(3, device=device).unsqueeze(0)
        if T is None:
            T = -center.view(1,3)

        cameras = OrthographicCameras(
            device=device,
            R=R,
            T=T,
            scale_xyz=((scale, scale, scale),)
        )
        return cameras

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
    
    def create_renderer(self, image_size, cameras, device="cuda"):
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras
            )
        )
        return renderer
    
    def render_screenshot(self, mesh, renderer):
        """
        Returns:
            image: (H,W,3)
        """
        image = renderer(mesh)[0, ..., :3]
        return image

    def run(self):
        points_cam = self._creat_point_cloud()
        print(f"Point cloud shape: {points_cam.shape}, device: {points_cam.device}")
        H, W = self.depth_img.shape[1:3]
        
        Xg = points_cam[:, 0].reshape(H, W)
        Yg = points_cam[:, 1].reshape(H, W)
        Zg = points_cam[:, 2].reshape(H, W)
        print(f"Xg shape: {Xg.shape}, Yg shape: {Yg.shape}, Zg shape: {Zg.shape}")
        verts, faces = self._build_dsm_mesh(Xg, Yg, Zg)
        verts_uvs, faces_uvs = self._build_uvs(H, W)
        mesh = self._attach_texture(verts, faces,verts_uvs, faces_uvs,orthophoto)
)
        # self._visualize_point_cloud(points_cam)

    
