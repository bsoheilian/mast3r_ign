import torch
import numpy as np
from typing import Tuple, Optional


class TexturedMesh3D:
    """
    A class to create and render 3D textured meshes from ground coordinate grids and RGB images.
    
    The class takes ground-based coordinate grids (Xg, Yg, Zg) and an RGB image, creates a 3D mesh,
    and provides methods to define camera geometries and render the mesh from various viewpoints.
    Initially designed for orthographic views but extensible to other camera models.
    """
    
    def __init__(self, Xg: torch.Tensor, Yg: torch.Tensor, Zg: torch.Tensor, rgb_image: np.ndarray):
        """
        Initialize the TexturedMesh3D class.
        
        Args:
            Xg: Ground X coordinates, torch.Tensor of shape (H, W), dtype=torch.float64
            Yg: Ground Y coordinates, torch.Tensor of shape (H, W), dtype=torch.float64
            Zg: Ground Z coordinates, torch.Tensor of shape (H, W), dtype=torch.float64
            rgb_image: RGB image, np.ndarray of shape (H, W, 3), dtype=float32
        
        Raises:
            RuntimeError: If CUDA is not available
        """
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This class requires CUDA to be installed and accessible.")
        
        # Set device to CUDA
        self.device = torch.device('cuda')
        
        # Move torch tensors to CUDA device
        Xg = Xg.to(self.device)
        Yg = Yg.to(self.device)
        Zg = Zg.to(self.device)
        
        # Convert rgb_image numpy array to torch tensor and move to CUDA
        if isinstance(rgb_image, np.ndarray):
            rgb_image = torch.from_numpy(rgb_image).float().to(self.device)
        else:
            rgb_image = rgb_image.to(self.device)
        
        assert Xg.dtype == torch.float64, "Xg must be torch.float64"
        assert Yg.dtype == torch.float64, "Yg must be torch.float64"
        assert Zg.dtype == torch.float64, "Zg must be torch.float64"
        assert rgb_image.dtype == torch.float32, "rgb_image must be torch.float32"
        
        H, W = Xg.shape
        assert Yg.shape == (H, W), "Yg shape must match Xg"
        assert Zg.shape == (H, W), "Zg shape must match Xg"
        assert rgb_image.shape == (H, W, 3), "rgb_image shape must be (H, W, 3)"
        
        self.H = H
        self.W = W
        self.Xg = Xg
        self.Yg = Yg
        self.Zg = Zg
        self.rgb_image = rgb_image
        
        # Create mesh vertices and faces
        self._create_mesh()
        
        # Camera geometry (to be set via set_camera_geometry)
        self.camera_geometry = None
    
    def _create_mesh(self):
        """
        Create mesh vertices and faces from coordinate grids.
        
        Vertices are created by stacking the ground coordinates.
        Faces are defined by connecting adjacent grid points into triangles.
        """
        # Flatten the coordinate grids to create vertices
        vertices_x = self.Xg.flatten()
        vertices_y = self.Yg.flatten()
        vertices_z = self.Zg.flatten()
        
        # Stack to create (N, 3) vertices
        self.vertices = torch.stack([vertices_x, vertices_y, vertices_z], dim=1)  # Shape: (H*W, 3)
        
        # Flatten RGB image to get vertex colors
        self.vertex_colors = self.rgb_image.reshape(-1, 3)  # Shape: (H*W, 3)
        
        # Create faces by connecting adjacent grid points
        faces = []
        for i in range(self.H - 1):
            for j in range(self.W - 1):
                # Get indices of the four corners of each grid cell
                top_left = i * self.W + j
                top_right = i * self.W + (j + 1)
                bottom_left = (i + 1) * self.W + j
                bottom_right = (i + 1) * self.W + (j + 1)
                
                # Create two triangles per grid cell
                faces.append([top_left, bottom_left, top_right])
                faces.append([top_right, bottom_left, bottom_right])
        
        self.faces = torch.tensor(faces, dtype=torch.long)  # Shape: (2*(H-1)*(W-1), 3)
    
    def set_camera_geometry(self, camera_type: str, **kwargs):
        """
        Set the camera geometry for rendering.
        
        Args:
            camera_type: Type of camera ('ortho' for orthographic, or other custom types)
            **kwargs: Camera-specific parameters
        """
        if camera_type == 'ortho':
            self.camera_geometry = self._create_ortho_camera(**kwargs)
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")
    
    def _create_ortho_camera(self, **kwargs) -> dict:
        """
        Create an orthographic camera geometry.
        
        Args:
            **kwargs: Optional parameters like scale, offset, etc.
        
        Returns:
            Dictionary containing orthographic camera parameters
        """
        camera = {
            'type': 'ortho',
            'params': kwargs
        }
        return camera
    
    def render(self, output_resolution: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Render the mesh using the current camera geometry.
        
        Args:
            output_resolution: Optional output resolution (H, W). Defaults to original image size.
        
        Returns:
            Rendered image as torch.Tensor of shape (H, W, 3), dtype=torch.float32
        """
        if self.camera_geometry is None:
            raise RuntimeError("Camera geometry not set. Call set_camera_geometry() first.")
        
        if output_resolution is None:
            output_resolution = (self.H, self.W)
        
        # Placeholder for rendering logic
        # This would typically involve:
        # 1. Projecting mesh vertices using camera geometry
        # 2. Rasterizing triangles
        # 3. Interpolating vertex colors
        
        rendered_image = torch.zeros(*output_resolution, 3, dtype=torch.float32)
        return rendered_image
    
    def get_vertices(self) -> torch.Tensor:
        """Get mesh vertices."""
        return self.vertices
    
    def get_faces(self) -> torch.Tensor:
        """Get mesh faces."""
        return self.faces
    
    def get_vertex_colors(self) -> torch.Tensor:
        """Get vertex colors from RGB image."""
        return self.vertex_colors
