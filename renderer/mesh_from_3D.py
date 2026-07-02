import math
import time

import torch
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.renderer import OrthographicCameras
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer,
    RasterizationSettings, SoftPhongShader, HardPhongShader, AmbientLights
)
import numpy as np
from typing import Tuple, Optional


class TexturedMesh3D:
    """
    A class to create and render 3D textured meshes from ground coordinate grids and RGB images.
    
    The class takes ground-based coordinate grids (Xg, Yg, Zg) and an RGB image, creates a 3D mesh,
    and provides methods to define camera geometries and render the mesh from various viewpoints.
    Initially designed for orthographic views but extensible to other camera models.
    """
    
    def __init__(
        self,
        Xg: torch.Tensor,
        Yg: torch.Tensor,
        Zg: torch.Tensor,
        rgb_image: np.ndarray,
        texture_sampling_mode: str = "bilinear",
    ):
        """
        Initialize the TexturedMesh3D class.
        
        Args:
            Xg: Ground X coordinates, torch.Tensor of shape (H, W), dtype=torch.float64
            Yg: Ground Y coordinates, torch.Tensor of shape (H, W), dtype=torch.float64
            Zg: Ground Z coordinates, torch.Tensor of shape (H, W), dtype=torch.float64
            rgb_image: RGB image, np.ndarray of shape (H, W, 3), dtype=float32
            texture_sampling_mode: Texture sampling for TexturesUV ("nearest" or "bilinear").
        
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
        # todo the type should already torch or ndarray otherwise transformed
        if isinstance(rgb_image, np.ndarray):
            rgb_image = torch.from_numpy(rgb_image).float().to(self.device)
            # Normalize rgb_image to [0, 1] range if values are in [0, 255]
            if rgb_image.max() > 1.0:
                rgb_image = rgb_image / 255.0
            # Keep channel-last layout (H, W, 3) for PyTorch3D TexturesUV maps
            
        else:
            raise TypeError("rgb_image must be a numpy ndarray")
        
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

        self.Xmin, self.Xmax = Xg.min().item(), Xg.max().item()
        self.Ymin, self.Ymax = Yg.min().item(), Yg.max().item()
        self.Zmin, self.Zmax = Zg.min().item(), Zg.max().item()
        self.Xinsertion = self.Xmin
        self.Yinsertion = self.Ymin 
        self.Zinsertion = 0.0

        # translate the double coordinates to float for rendering by subtracting Xmin and Ymin
        self.Xg = (self.Xg - self.Xmin).float()
        self.Yg = (self.Yg - self.Ymin).float() 
        self.Zg = self.Zg.float()
        self.Xmin, self.Xmax = 0, self.Xmax - self.Xmin
        self.Ymin, self.Ymax = 0, self.Ymax - self.Ymin
        
        
        self.rgb_image = rgb_image
        if texture_sampling_mode not in ("nearest", "bilinear"):
            raise ValueError("texture_sampling_mode must be 'nearest' or 'bilinear'")
        self.texture_sampling_mode = texture_sampling_mode
        
        # Create mesh vertices and faces
        self._create_mesh_texture_indexes()

        # Create textured mesh
        self._create_textured_mesh()
        

    def _sync_cuda(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
    
    def _create_mesh_texture_indexes(self, verbose: bool=False):
        """
        Create mesh vertices and faces from coordinate grids.
        
        Vertices are created by stacking the ground coordinates.
        Faces are defined by connecting adjacent grid points into triangles.
        """
        # Flatten the coordinate grids to create vertices
        vertices_x = self.Xg.flatten()
        vertices_y = self.Yg.flatten()
        vertices_z = self.Zg.flatten()
        
        # Stack to create (N, 3) vertices and convert to float32 for rendering compatibility
        self.vertices = torch.stack([vertices_x, vertices_y, vertices_z], dim=1).float()  # Shape: (H*W, 3), dtype: float32
        
        xs = torch.linspace(0, 1, self.W, device=self.device)
        ys = torch.linspace(1, 0, self.H, device=self.device)  # flip Y for image coords
        u, v = torch.meshgrid(xs, ys, indexing="xy")
        self.verts_uvs = torch.stack([u, v], dim=-1).reshape(-1, 2)

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
        
        self.faces = torch.tensor(faces, dtype=torch.long, device=self.device)  # Shape: (2*(H-1)*(W-1), 3)
        if verbose:
            print(f"[_create_mesh_texture_indexes] Created mesh with {self.vertices.shape[0]} vertices and {self.faces.shape[0]} faces.")
            print(f"[_create_mesh_texture_indexes] Vertices dtype: {self.vertices.dtype}, Faces dtype: {self.faces.dtype}")
    
    def _create_textured_mesh(self):
        textures = TexturesUV(
            maps=self.rgb_image.unsqueeze(0),  # (1, H, W, 3)
            verts_uvs=self.verts_uvs.unsqueeze(0),
            faces_uvs=self.faces.unsqueeze(0),
            sampling_mode=self.texture_sampling_mode,
        )
        self.mesh = Meshes(verts=[self.vertices], faces=[self.faces], textures=textures)

    def _get_ortho_camera(self, gsd: float=0.1, verbose: bool=False):
        # C is the coordinate of the camera center in world coordinates. 
        # For orthographic camera, we can place it above the mesh.
        cx = (self.Xmin + self.Xmax) / 2
        cy = (self.Ymin + self.Ymax) / 2
        cz = self.Zmin + 10.  # Place camera above the mesh 
        C = torch.tensor([cx, cy, cz], device=self.device, dtype=torch.float32)

        # Define orthographic camera Rotation matrix 
        # check the docs here to understand the rotation matrix for orthographic camera:
        # https://pytorch3d.org/docs/cameras
        # Xcam and Ycam axes are parallel to Xg and Yg axes, and Z axis is pointing downwards (towards the mesh)
        # Xcam is in opposite direction of Xg, 
        # Ycam is in direction of Yg, 
        # Zcam is in opposite direction of Zg
        R = torch.tensor([[-1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., -1.]], device=self.device, dtype=torch.float32)
        
        # the transformes are applied as RXg + T,  
        # intuitially if a point is exactly beneath the camera Z axis
        # its camera coordinates should be (0, 0, Zcam) where Zcam 
        # so  0 = RXg + T => T = -RXg, so the translation vector is -R @ C
        T = -R @ C
        
        #final R and T
        R = R.unsqueeze(0)  # Add batch dimension
        T = T.unsqueeze(0)  # Add batch dimension
        
        # scale is defined to scale the mesh to fit in the [-1, 1] range for rendering.
        scale = 2.0/(max(self.Xmax - self.Xmin, self.Ymax - self.Ymin))  # Scale to fit in [-1, 1] range

        # PPA no translation in x and y
        px = py = 0.0  
        
        #image size is defined based on the GSD and the mesh extents
        W = math.ceil((self.Xmax - self.Xmin) / gsd)
        H = math.ceil((self.Ymax - self.Ymin) / gsd)

        if verbose:
            print(f"[get_ortho_camera] gsd={gsd}, image_size=({H}, {W}), scale={scale}, px={px}, py={py}")
            print(f"[get_ortho_camera] R:\n{R}\nT:\n{T}\nC:\n{C}")

        return OrthographicCameras(device=self.device, R=R, T=T,
                                   focal_length=((scale, scale),),
                                   principal_point=((px,py),),
                                   image_size=((H, W),)
                                   )

    def _create_ortho_renderer(
        self,
        image_size,
        camera,
        bin_size: Optional[int] = None,
        max_faces_per_bin: Optional[int] = None,
        faces_per_pixel: int = 1,
        blur_radius: float = 0.0,
        cull_backfaces: bool = False,
        use_hard_shader: bool = True,
        verbose: bool = False,
    ):
        raster = RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,                #default : 0.0
            faces_per_pixel=faces_per_pixel,        #default : 1
            bin_size=bin_size,                      #default : None, use pytorch3d heuristic
            max_faces_per_bin=max_faces_per_bin,    #default : None, use pytorch3d heuristic
            cull_backfaces=cull_backfaces,          #default : False, if True, drops back-facing triangles (often removes speckles on noisy meshes)
        )

        if verbose:
            print(f"Creating orthographic renderer with the following settings:")
            print(f"image_size              = {image_size}")
            print(f"blur_radius             = {blur_radius}")
            print(f"faces_per_pixel         = {faces_per_pixel}")
            print(f"bin_size                = {bin_size}")
            print(f"max_faces_per_bin       = {max_faces_per_bin}")
            print(f"cull_backfaces          = {cull_backfaces}")
            print(f"use_hard_shader         = {use_hard_shader}")

        device = camera.device
        # Full ambient light so texture colors are rendered as-is without shading
        lights = AmbientLights(device=device)

        shader_cls = HardPhongShader if use_hard_shader else SoftPhongShader

        return MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=raster
            ),
            shader=shader_cls(device=device, cameras=camera, lights=lights)
        )
    
    def _render_ortho(self, mesh, renderer):
        img = renderer(mesh)[0, ..., :3]
        return img
    
    
    def create_orth(
        self,
        gsd: float=0.1,
        profile: bool=False,
        bin_size: Optional[int]=None,
        max_faces_per_bin: Optional[int]=None,
        safe_raster: bool=True,
        faces_per_pixel: int=1,
        blur_radius: float=0.0,
        cull_backfaces: bool=True, 
        use_hard_shader: bool=True, 
        verbose: bool=False,
    ):
        """
        Create an orthographic view of the textured mesh.
        
        Args:
            gsd: Ground Sample Distance (GSD) in world units per pixel.
            show: If True, display the rendered image using matplotlib.
            profile: If True, prints step timings and a torch profiler summary. if activated show shall be set to False.
            bin_size: Rasterization bin size. Use None for PyTorch3D heuristic or 0 for no binning (much slower but avoids overflow).
            max_faces_per_bin: Upper bound for coarse raster bins; increase this when overflow warnings appear.
            safe_raster: If True (and both bin_size and max_faces_per_bin are None), auto-sets a conservative max_faces_per_bin to reduce coarse-bin overflow risk while keeping PyTorch3D binning.
            faces_per_pixel: Number of faces stored per pixel (1 is sharpest/fastest).
            blur_radius: Raster blur radius in NDC; keep 0 for crisp edges. very small values (e.g., 1e-10 to 1e-30) can help avoid artifacts in some cases.
            cull_backfaces: If True, drops back-facing triangles (often removes speckles on noisy meshes).
            use_hard_shader: If True, uses HardPhongShader for crisper results; if False, uses SoftPhongShader for smoother results.
        
        Returns:
            img: Rendered orthographic image as a torch.Tensor of shape (H, W, 3).
        """


        t0 = time.perf_counter()
        camera = self._get_ortho_camera(gsd=gsd, verbose=verbose)
        if verbose: self._sync_cuda()
        t1 = time.perf_counter()

        # camera.image_size[0] returns a tensor, convert to tuple of ints for RasterizationSettings
        image_size_tensor = camera.image_size[0]
        image_size = tuple(int(x.item() if isinstance(x, torch.Tensor) else x) for x in image_size_tensor)

        n_faces = int(self.faces.shape[0])

        effective_bin_size = bin_size
        effective_max_faces_per_bin = max_faces_per_bin
        if safe_raster and effective_bin_size is None and effective_max_faces_per_bin is None:
            # Auto-compute max_faces_per_bin so coarse bins never overflow.
            # PyTorch3D default bin_size heuristic: ceil(sqrt(image_max_dim / 2))
            # Each bin covers bin_size x bin_size pixels; estimate how many faces fall in a bin.
            # img_max = max(image_size)
            auto_bin = max(1, math.ceil(math.sqrt(max(image_size) / 2)))
            # upper-bound: assume all faces could land in one bin (conservative but correct)
            effective_max_faces_per_bin = max(n_faces, 30000)
            if verbose:
                print(f"[create_orth] auto bin_size (estimation)={auto_bin}, "
                      f"max_faces_per_bin={effective_max_faces_per_bin} (from {n_faces:,} faces)")

        renderer = self._create_ortho_renderer(
            image_size=image_size,
            camera=camera,
            bin_size=effective_bin_size,
            max_faces_per_bin=effective_max_faces_per_bin,
            faces_per_pixel=faces_per_pixel,
            blur_radius=blur_radius,
            cull_backfaces=cull_backfaces,
            use_hard_shader=use_hard_shader,
            verbose=verbose,
        )
        if verbose: self._sync_cuda()
        t2 = time.perf_counter()

        img = self._render_ortho(self.mesh, renderer)
        if verbose: self._sync_cuda()
        t3 = time.perf_counter()
        
        if verbose:
            n_verts = int(self.vertices.shape[0])
            n_pixels = int(image_size[0] * image_size[1])
            print(
                f"[create_orth] image_size={image_size} ({n_pixels/1e6:.2f} MP), "
                f"verts={n_verts:,}, faces={n_faces:,}, gsd={gsd}, "
                f"bin_size={effective_bin_size}, max_faces_per_bin={effective_max_faces_per_bin}"
            )
            print(
                f"[create_orth] camera={t1-t0:.3f}s, renderer_build={t2-t1:.3f}s, render={t3-t2:.3f}s, total={t3-t0:.3f}s"
            )
        return img









  
    
