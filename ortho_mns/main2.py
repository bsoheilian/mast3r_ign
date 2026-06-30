import torch
import numpy as np
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

# ============================================
# Step 1: Backproject Depth to 3D Point Cloud
# ============================================

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


# ============================================
# Step 2: Rasterize 3D Points to DSM Grid
# ============================================

def rasterize_to_dsm(points_world, rgb, xmin, xmax, ymin, ymax, rows, cols, device='cuda'):
    """
    Rasterize 3D points to DSM and orthoimage grids.
    
    Args:
        points_world: (H, W, 3) tensor, 3D points in world coordinates.
        rgb: (H, W, 3) tensor, RGB values for each pixel.
        xmin, xmax, ymin, ymax: float, ground bounds in world coordinates.
        rows, cols: int, resolution of the DSM/ortho grid.
        device: str, device to use for computation.
    
    Returns:
        dsm: (rows, cols) tensor, Digital Surface Model.
        ortho: (rows, cols, 3) tensor, orthoimage.
    """
    x = points_world[..., 0]
    y = points_world[..., 1]
    z = points_world[..., 2]
    
    # Scale to grid indices
    x_grid = ((x - xmin) / (xmax - xmin) * (cols - 1)).long()
    # y_grid = ((y - ymin) / (ymax - ymin) * (rows - 1)).long()
    y_grid = ((ymax - y) / (ymax - ymin) * (rows - 1)).long()

    # Clamp to grid bounds
    mask = (x_grid >= 0) & (x_grid < cols) & (y_grid >= 0) & (y_grid < rows)
    x_grid, y_grid, z = x_grid[mask], y_grid[mask], z[mask]
    rgb = rgb[mask]
    
    # Initialize DSM and ortho grids
    dsm = torch.full((rows, cols), -torch.inf, device=device)
    ortho = torch.zeros((rows, cols, 3), device=device)
    
    # Scatter max Z and average RGB
    dsm[y_grid, x_grid] = torch.maximum(dsm[y_grid, x_grid], z)
    ortho[y_grid, x_grid] += rgb.float()
    count = torch.zeros((rows, cols), device=device)
    count[y_grid, x_grid] += 1
    ortho = ortho / count.unsqueeze(-1).clamp(min=1)
    
    return dsm, ortho


# ============================================
# Step 3: Render Textured Surface with Orthogonal Camera
# ============================================

def render_with_pytorch3d(dsm, ortho, xmin, xmax, ymin, ymax, rows, cols, device='cuda'):
    """
    Render the DSM as a textured mesh using PyTorch3D and an orthogonal camera.

    Args:
        dsm: (rows, cols) tensor, Digital Surface Model.
        ortho: (rows, cols, 3) tensor, orthoimage.
        xmin, xmax, ymin, ymax: float, ground bounds in world coordinates.
        rows, cols: int, resolution of the DSM/ortho grid.
        device: str, device to use for computation.

    Returns:
        rendered_image: (rows, cols, 3) tensor, rendered image.
    """
    # Convert DSM to mesh vertices
    x = torch.linspace(xmin, xmax, cols, device=device)
    y = torch.linspace(ymin, ymax, rows, device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
    verts = torch.stack([x_grid, y_grid, dsm], dim=-1).reshape(-1, 3)  # (rows*cols, 3)

    # Create faces (grid connectivity)
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = i * cols + j
            v1 = i * cols + j + 1
            v2 = (i + 1) * cols + j
            v3 = (i + 1) * cols + j + 1
            faces.extend([[v0, v1, v2], [v1, v3, v2]])
    faces = torch.tensor(faces, device=device)

    # Textures from orthoimage
    tex_verts = ortho.reshape(-1, 3)

    # Create mesh with textures
    mesh = Meshes(
        verts=[verts],
        faces=[faces],
        textures=TexturesVertex(verts_features=[tex_verts])
    ).to(device)

    # Orthogonal camera (Z-up, looking down)
    R = torch.eye(3, device=device)
    R[0, 0] = 0  # X-axis points right
    R[0, 1] = 1
    R[1, 0] = -1
    R[1, 1] = 0
    T = torch.tensor([(xmin + xmax)/2, (ymin + ymax)/2, 0], device=device)
    camera = OrthographicCameras(
        R=R.unsqueeze(0),
        T=T.unsqueeze(0),
        device=device
    )

    # Rasterize
    raster_settings = RasterizationSettings(image_size=(rows, cols))
    rasterizer = MeshRasterizer(camera, raster_settings)

    # Get fragments
    fragments = rasterizer(mesh)

    # Extract the rendered image from fragments
    # Use the textured mesh to get the color for each pixel
    texels = mesh.textures.verts_features_packed()
    pix_to_face = fragments.pix_to_face.squeeze(0)  # (H, W)
    bary_coords = fragments.bary_coords.squeeze(0)  # (H, W, 3)

    # Get the vertex indices for each face
    verts_packed = mesh.verts_packed()
    faces_packed = mesh.faces_packed()
    face_verts = verts_packed[faces_packed]  # (F, 3, 3)

    # Compute the interpolated texture coordinates
    # For each pixel, get the barycentric coordinates and interpolate the texture
    # This is a simplified approach; for full correctness, use a shader or post-processing
    # Here, we assume the texture is per-vertex and interpolate it
    face_textures = texels[faces_packed]  # (F, 3, 3)
    interpolated_textures = torch.einsum('fvc,hwf,v->hwc', face_textures, bary_coords, torch.ones(3, device=device))
    interpolated_textures = interpolated_textures / bary_coords.sum(dim=-1, keepdim=True)

    # Assign textures to pixels
    rendered_image = torch.zeros((rows, cols, 3), device=device)
    mask = pix_to_face >= 0
    rendered_image[mask] = interpolated_textures[pix_to_face[mask]]

    return rendered_image

def render_orthogonal(ortho, dsm, xmin, xmax, ymin, ymax, pixel_size, device='cuda'):
    """
    Render orthogonal view using grid sampling (lightweight approach).
    
    Args:
        ortho: (rows, cols, 3) tensor, orthoimage.
        dsm: (rows, cols) tensor, Digital Surface Model.
        xmin, xmax, ymin, ymax: float, ground bounds in world coordinates.
        pixel_size: float, size of each pixel in world coordinates.
        device: str, device to use for computation.
    
    Returns:
        orthogonal_image: (rows, cols, 3) tensor, rendered orthogonal image.
    """
    H, W = ortho.shape[:2]
    y = torch.linspace(ymin, ymax, H, device=device)
    x = torch.linspace(xmin, xmax, W, device=device)
    y, x = torch.meshgrid(y, x, indexing='ij')
    
    # Sample orthoimage at these coordinates
    grid = torch.stack([x, y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    grid = (grid - torch.tensor([xmin, ymin], device=device)) / torch.tensor([xmax - xmin, ymax - ymin], device=device) * 2 - 1  # Normalize to [-1, 1]
    orthogonal_image = torch.nn.functional.grid_sample(
        ortho.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).permute(1, 2, 0)
    
    return orthogonal_image


# ============================================
# Full Pipeline
# ============================================

def read_depth_image(file_path, width:int):
    data = np.load(file_path)
    depth_image = data.reshape(-1, width)
    return depth_image

def read_rgb_image(file_path):
    data = np.load(file_path)
    return data

def main():
    # Example usage: Replace with your actual data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # device = 'cpu'
    # Load your data here (replace with actual data)
    # depth: (H, W) numpy array or tensor
    # rgb: (H, W, 3) numpy array or tensor
    # K: (3, 3) camera intrinsics matrix
    # R: (3, 3) rotation matrix
    # t: (3,) translation vector
    # xmin, xmax, ymin, ymax: ground bounds in world coordinates
    # rows, cols: resolution of the DSM/ortho grid
    
    # depth = torch.rand(480, 640, device=device) * 10  # Example depth map
    # depth = 
    # rgb = torch.rand(480, 640, 3, device=device)  # Example RGB image
    rgb = read_rgb_image("./output/debug_rgb_Lille-150127_0485-11-00002_0000384.jpg_2.npy")
    print(f"RGB image shape: {rgb.shape}, dtype: {rgb.dtype}")
    rgb = torch.from_numpy(rgb)
    rgb = rgb.to(device)
    
    depth = read_depth_image("./output/debug_depthmap_Lille-150127_0485-11-00002_0000384.jpg_2.npy", rgb.shape[1])
    print(f"Depth image shape: {depth.shape}, dtype: {depth.dtype}")

    depth = torch.from_numpy(depth)
    # K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=torch.float32, device=device)
    R = torch.eye(3, device=device)
    t = torch.tensor([0, 0, 0], device=device)

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

    
    xmin, xmax, ymin, ymax = -10, 10, -10, 10
    rows, cols = 200, 200
    
    # Step 1: Generate 3D point cloud in world coordinates
    points_world = depth_to_3d(depth, K, R, t, device)
    
     
    # Step 2: Rasterize to DSM and orthoimage
    dsm, ortho = rasterize_to_dsm(points_world, rgb, xmin, xmax, ymin, ymax, rows, cols, device)
    
     
    rendered_image_pytorch3d = None
    # if HAS_PYTORCH3D:
    #     # Step 3: Render with PyTorch3D
    # rendered_image_pytorch3d = render_with_pytorch3d(dsm, ortho, xmin, xmax, ymin, ymax, rows, cols, device)
    # else:
    #     print("PyTorch3D is not installed; skipping PyTorch3D rendering.")
    
    # Step 3 (alternative): Render with grid sampling
    rendered_image_grid = render_orthogonal(ortho, dsm, xmin, xmax, ymin, ymax, 0.1, device)
    
    print("DSM shape:", dsm.shape)
    print("Orthoimage shape:", ortho.shape)
    # if rendered_image_pytorch3d is not None:
    #     print("Rendered image (PyTorch3D) shape:", rendered_image_pytorch3d.shape)
    print("Rendered image (Grid) shape:", rendered_image_grid.shape)

    # Display rendered_image_grid
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    if rendered_image_grid.dim() == 3:
        if rendered_image_grid.shape[0] == 3:
            plt.imshow(rendered_image_grid.permute(1, 2, 0).cpu().numpy())
        elif rendered_image_grid.shape[2] == 3:
            plt.imshow(rendered_image_grid.cpu().numpy())
    else:
        plt.imshow(rendered_image_grid.cpu().numpy(), cmap='gray')
    plt.title("Rendered Orthoimage (Grid)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return

if __name__ == "__main__":
    main()