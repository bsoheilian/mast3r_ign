from dataclasses import dataclass
import numpy as np
from typing import Union
from pathlib import Path
import torch


@dataclass
class Orientation:
    """Dataclass to represent camera orientation with intrinsics and extrinsics."""
    
    _K: np.ndarray  # Intrinsics matrix (3x3, floats)
    _R: np.ndarray  # Rotation matrix (3x3, doubles)
    _C: np.ndarray  # 3D translation vector (3x1, floats)
    
    def __post_init__(self):
        """Validate matrix dimensions and types after initialization."""
        self._validate_matrices()
    
    def _validate_matrices(self):
        """Check that matrices have correct shapes and types."""
        if self._K.shape != (3, 3):
            raise ValueError(f"K matrix must be 3x3, got shape {self._K.shape}")
        if self._K.dtype != np.float32 and self._K.dtype != np.float64:
            raise ValueError(f"K matrix must be float type, got {self._K.dtype}")
        
        if self._R.shape != (3, 3):
            raise ValueError(f"R matrix must be 3x3, got shape {self._R.shape}")
        if self._R.dtype != np.float64:
            raise ValueError(f"R matrix must be double (float64), got {self._R.dtype}")
        
        if self._C.shape != (3, 1) and self._C.shape != (3,):
            raise ValueError(f"C vector must be 3x1 or (3,), got shape {self._C.shape}")
        if self._C.dtype != np.float32 and self._C.dtype != np.float64:
            raise ValueError(f"C vector must be float type, got {self._C.dtype}")
    
    @classmethod
    def from_arrays(cls, K: np.ndarray, R: np.ndarray, C: np.ndarray) -> "Orientation":
        """
        Create Orientation from numpy arrays.
        
        Args:
            K: Intrinsics matrix (3x3, float)
            R: Rotation matrix (3x3, double/float64)
            C: 3D translation vector (3x1 or (3,), float)
        
        Returns:
            Orientation instance
        
        Raises:
            ValueError: If matrix dimensions or types are incorrect
        """
        return cls(_K=K, _R=R, _C=C)
    
    @classmethod
    def from_files(cls, K_path: Union[str, Path], R_path: Union[str, Path], C_path: Union[str, Path]) -> "Orientation":
        """
        Create Orientation from text files containing numpy arrays.
        
        Args:
            K_path: Path to file containing K matrix
            R_path: Path to file containing R matrix
            C_path: Path to file containing C vector
        
        Returns:
            Orientation instance
        
        Raises:
            ValueError: If matrix dimensions or types are incorrect
            FileNotFoundError: If any file does not exist
        """
        K = np.loadtxt(K_path, dtype=np.float32)
        R = np.loadtxt(R_path, dtype=np.float64)
        C = np.loadtxt(C_path, dtype=np.float32)
        
        return cls(K, R.T, C) # Note: R is transposed to match the expected orientation
        # to do : make it possible to provide R and T in both directions (cam2world and world2cam) and handle accordingly
    
    def apply_K_torch(self, depth_img: np.ndarray, device: Union[str, torch.device] = "cuda"):
        """
        Apply intrinsics matrix K to a depth image to convert pixel coordinates to camera coordinates using PyTorch.
        
        Args:
            depth_img: Depth image as a 2D numpy array (H, W)
        
        Returns:
            Xcam, Ycam, Zcam: Camera coordinates as 2D numpy arrays (H, W)
        """
        H, W = depth_img.shape
        u = torch.arange(W, device=device, dtype=torch.float64) # use float64 for better precision
        v = torch.arange(H, device=device, dtype=torch.float64)
        u, v = torch.meshgrid(u, v, indexing='xy')  # u: (H, W), v: (H, W)  
        depth = torch.from_numpy(depth_img).to(device=device, dtype=torch.float64)
        Xcam, Ycam, Zcam = self.apply_K(u, v, depth, direction="img2cam")
        return Xcam, Ycam, Zcam

    
    def apply_K(self,
                x: Union[np.ndarray, torch.Tensor, float],
                y: Union[np.ndarray, torch.Tensor, float],
                z: Union[np.ndarray, torch.Tensor, float] = 1.0,
                direction: str = "img2cam") -> tuple:
        """
        Apply intrinsics matrix K in either direction.

        Args:
            x: u pixel coordinate(s) if img2cam, Xcam if cam2img
            y: v pixel coordinate(s) if img2cam, Ycam if cam2img
            z: depth if img2cam, Zcam if cam2img (default 1.0)
            direction: "img2cam" (pixel+depth -> camera coords)
                       "cam2img" (camera coords -> pixel+depth)

        Returns:
            "img2cam": (Xcam, Ycam, Zcam)
            "cam2img": (u, v, depth)
        """
        if direction not in ("img2cam", "cam2img"):
            raise ValueError(f"direction must be 'img2cam' or 'cam2img', got '{direction}'")

        is_torch = isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor) or isinstance(z, torch.Tensor)

        if is_torch:
            x = torch.as_tensor(x)
            y = torch.as_tensor(y, device=x.device, dtype=x.dtype)
            z = torch.as_tensor(z, device=x.device, dtype=x.dtype)
            x, y, z = torch.broadcast_tensors(x, y, z)
            orig_shape = x.shape

            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)
            z_flat = z.reshape(-1)
            K = torch.from_numpy(self._K).to(device=x.device, dtype=x.dtype)

            if direction == "img2cam":
                pixels = torch.stack([x_flat, y_flat, torch.ones_like(x_flat)], dim=0)
                normalized = torch.linalg.inv(K) @ pixels
                return (
                    (normalized[0] * z_flat).reshape(orig_shape),
                    (normalized[1] * z_flat).reshape(orig_shape),
                    (normalized[2] * z_flat).reshape(orig_shape),
                )
            points = torch.stack([x_flat, y_flat, z_flat], dim=0)
            projected = K @ points
            return (
                (projected[0] / projected[2]).reshape(orig_shape),
                (projected[1] / projected[2]).reshape(orig_shape),
                projected[2].reshape(orig_shape),
            )

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        x, y, z = np.broadcast_arrays(x, y, z)
        orig_shape = x.shape

        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        z_flat = z.reshape(-1)

        if direction == "img2cam":
            pixels = np.stack([x_flat, y_flat, np.ones_like(x_flat)], axis=0)
            normalized = np.linalg.inv(self._K) @ pixels
            return (
                (normalized[0] * z_flat).reshape(orig_shape),
                (normalized[1] * z_flat).reshape(orig_shape),
                (normalized[2] * z_flat).reshape(orig_shape),
            )
        points = np.stack([x_flat, y_flat, z_flat], axis=0)
        projected = self._K @ points
        return (
            (projected[0] / projected[2]).reshape(orig_shape),
            (projected[1] / projected[2]).reshape(orig_shape),
            projected[2].reshape(orig_shape),
        )

    def apply_ext(self,
                  x: Union[np.ndarray, torch.Tensor, float],
                  y: Union[np.ndarray, torch.Tensor, float],
                  z: Union[np.ndarray, torch.Tensor, float],
                  direction: str = "cam2world") -> tuple:
        """
        Apply extrinsics (R, C) in either direction.

        Args:
            x: Xcam if cam2world, Xworld if world2cam
            y: Ycam if cam2world, Yworld if world2cam
            z: Zcam if cam2world, Zworld if world2cam
            direction: "cam2world" or "world2cam"

        Returns:
            "cam2world": (Xworld, Yworld, Zworld)
            "world2cam": (Xcam, Ycam, Zcam)
        """
        if direction not in ("cam2world", "world2cam"):
            raise ValueError(f"direction must be 'cam2world' or 'world2cam', got '{direction}'")

        is_torch = isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor) or isinstance(z, torch.Tensor)

        if is_torch:
            x = torch.as_tensor(x)
            y = torch.as_tensor(y, device=x.device, dtype=x.dtype)
            z = torch.as_tensor(z, device=x.device, dtype=x.dtype)
            x, y, z = torch.broadcast_tensors(x, y, z)
            orig_shape = x.shape

            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)
            z_flat = z.reshape(-1)

            points = torch.stack([x_flat, y_flat, z_flat], dim=0)
            R = torch.from_numpy(self._R).to(device=x.device, dtype=x.dtype)
            C = torch.from_numpy(np.asarray(self._C).reshape(3)).to(device=x.device, dtype=x.dtype).reshape(3, 1)

            if direction == "cam2world":
                out = torch.linalg.inv(R) @ points + C
            else:
                out = R @ (points - C)
            return (out[0].reshape(orig_shape), out[1].reshape(orig_shape), out[2].reshape(orig_shape))

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        x, y, z = np.broadcast_arrays(x, y, z)
        orig_shape = x.shape

        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        z_flat = z.reshape(-1)

        points = np.stack([x_flat, y_flat, z_flat], axis=0)
        C = np.asarray(self._C).reshape(3, 1)

        if direction == "cam2world":
            out = np.linalg.inv(self._R) @ points + C
        else:
            out = self._R @ (points - C)
        return (out[0].reshape(orig_shape), out[1].reshape(orig_shape), out[2].reshape(orig_shape))


if __name__ == "__main__":
    # Create test matrices
    print("Testing Orientation class...")
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    R = np.array([
        [0.9998, -0.0175, 0.0090],
        [0.0176, 0.9998, -0.0045],
        [-0.0089, 0.0047, 0.9999]
    ], dtype=np.float64)

    C = np.array([1.2, -0.8, 0.5], dtype=np.float32)
    
    # Initialize Orientation with ndarrays
    ori = Orientation.from_arrays(K, R, C)
    print("Orientation initialized successfully")
    
    # Round-trip A with numpy inputs: img -> cam -> world -> cam -> img
    u_np = np.array([320.0, 640.0])
    v_np = np.array([240.0, 480.0])
    depth_np = np.array([2.0, 3.0])

    Xcam_np, Ycam_np, Zcam_np = ori.apply_K(u_np, v_np, depth_np, direction="img2cam")
    Xworld_np, Yworld_np, Zworld_np = ori.apply_ext(Xcam_np, Ycam_np, Zcam_np, direction="cam2world")
    Xcam_np_rt, Ycam_np_rt, Zcam_np_rt = ori.apply_ext(Xworld_np, Yworld_np, Zworld_np, direction="world2cam")
    u_np_rt, v_np_rt, depth_np_rt = ori.apply_K(Xcam_np_rt, Ycam_np_rt, Zcam_np_rt, direction="cam2img")

    du_np = np.max(np.abs(u_np_rt - u_np))
    dv_np = np.max(np.abs(v_np_rt - v_np))
    dd_np = np.max(np.abs(depth_np_rt - depth_np))
    dXcam_np_rt = np.max(np.abs(Xcam_np_rt - Xcam_np))
    dYcam_np_rt = np.max(np.abs(Ycam_np_rt - Ycam_np))
    dZcam_np_rt = np.max(np.abs(Zcam_np_rt - Zcam_np))
    print(f"Numpy round-trip A max abs diff: du={du_np}, dv={dv_np}, ddepth={dd_np}, dXcam={dXcam_np_rt}, dYcam={dYcam_np_rt}, dZcam={dZcam_np_rt}")
    assert np.allclose(Xcam_np_rt, Xcam_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(Ycam_np_rt, Ycam_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(Zcam_np_rt, Zcam_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(u_np_rt, u_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(v_np_rt, v_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(depth_np_rt, depth_np, rtol=1e-6, atol=1e-6)

    # Round-trip B with numpy inputs: world -> cam -> img -> cam -> world
    Xworld_seed_np = np.array([1.0, 2.0])
    Yworld_seed_np = np.array([0.5, -1.0])
    Zworld_seed_np = np.array([6.0, 8.0])

    Xcam_from_world_np, Ycam_from_world_np, Zcam_from_world_np = ori.apply_ext(
        Xworld_seed_np, Yworld_seed_np, Zworld_seed_np, direction="world2cam"
    )
    u_from_world_np, v_from_world_np, depth_from_world_np = ori.apply_K(
        Xcam_from_world_np, Ycam_from_world_np, Zcam_from_world_np, direction="cam2img"
    )
    Xcam_from_img_np, Ycam_from_img_np, Zcam_from_img_np = ori.apply_K(
        u_from_world_np, v_from_world_np, depth_from_world_np, direction="img2cam"
    )
    Xworld_back_np, Yworld_back_np, Zworld_back_np = ori.apply_ext(
        Xcam_from_img_np, Ycam_from_img_np, Zcam_from_img_np, direction="cam2world"
    )

    dXw_np = np.max(np.abs(Xworld_back_np - Xworld_seed_np))
    dYw_np = np.max(np.abs(Yworld_back_np - Yworld_seed_np))
    dZw_np = np.max(np.abs(Zworld_back_np - Zworld_seed_np))
    print(f"Numpy round-trip B max abs diff: dXworld={dXw_np}, dYworld={dYw_np}, dZworld={dZw_np}")
    assert np.allclose(Xworld_back_np, Xworld_seed_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(Yworld_back_np, Yworld_seed_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(Zworld_back_np, Zworld_seed_np, rtol=1e-6, atol=1e-6)
    
    # Round-trip tests with torch CUDA inputs + numpy-vs-torch checks at each step
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Use float64 to make strict 1e-6 round-trip checks numerically stable on CUDA.
        u_torch = torch.tensor([320.0, 640.0], device=device, dtype=torch.float64)
        v_torch = torch.tensor([240.0, 480.0], device=device, dtype=torch.float64)
        depth_torch = torch.tensor([2.0, 3.0], device=device, dtype=torch.float64)

        Xcam_torch, Ycam_torch, Zcam_torch = ori.apply_K(u_torch, v_torch, depth_torch, direction="img2cam")
        Xworld_torch, Yworld_torch, Zworld_torch = ori.apply_ext(Xcam_torch, Ycam_torch, Zcam_torch, direction="cam2world")
        Xcam_torch_rt, Ycam_torch_rt, Zcam_torch_rt = ori.apply_ext(Xworld_torch, Yworld_torch, Zworld_torch, direction="world2cam")
        u_torch_rt, v_torch_rt, depth_torch_rt = ori.apply_K(Xcam_torch_rt, Ycam_torch_rt, Zcam_torch_rt, direction="cam2img")

        du_torch = torch.max(torch.abs(u_torch_rt - u_torch)).item()
        dv_torch = torch.max(torch.abs(v_torch_rt - v_torch)).item()
        dd_torch = torch.max(torch.abs(depth_torch_rt - depth_torch)).item()
        dXcam_torch_rt = torch.max(torch.abs(Xcam_torch_rt - Xcam_torch)).item()
        dYcam_torch_rt = torch.max(torch.abs(Ycam_torch_rt - Ycam_torch)).item()
        dZcam_torch_rt = torch.max(torch.abs(Zcam_torch_rt - Zcam_torch)).item()
        print(f"Torch CUDA round-trip A max abs diff: du={du_torch}, dv={dv_torch}, ddepth={dd_torch}, dXcam={dXcam_torch_rt}, dYcam={dYcam_torch_rt}, dZcam={dZcam_torch_rt}")

        Xcam_torch_np = Xcam_torch.detach().cpu().numpy()
        Ycam_torch_np = Ycam_torch.detach().cpu().numpy()
        Zcam_torch_np = Zcam_torch.detach().cpu().numpy()
        Xworld_torch_np = Xworld_torch.detach().cpu().numpy()
        Yworld_torch_np = Yworld_torch.detach().cpu().numpy()
        Zworld_torch_np = Zworld_torch.detach().cpu().numpy()
        Xcam_torch_rt_np = Xcam_torch_rt.detach().cpu().numpy()
        Ycam_torch_rt_np = Ycam_torch_rt.detach().cpu().numpy()
        Zcam_torch_rt_np = Zcam_torch_rt.detach().cpu().numpy()
        u_torch_rt_np = u_torch_rt.detach().cpu().numpy()
        v_torch_rt_np = v_torch_rt.detach().cpu().numpy()
        depth_torch_rt_np = depth_torch_rt.detach().cpu().numpy()

        dX_np_torch = np.max(np.abs(Xcam_np - Xcam_torch_np))
        dY_np_torch = np.max(np.abs(Ycam_np - Ycam_torch_np))
        dZ_np_torch = np.max(np.abs(Zcam_np - Zcam_torch_np))
        print(f"Img2cam numpy-vs-torch max abs diff: dX={dX_np_torch}, dY={dY_np_torch}, dZ={dZ_np_torch}")
        assert np.allclose(Xcam_np, Xcam_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Ycam_np, Ycam_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Zcam_np, Zcam_torch_np, rtol=1e-6, atol=1e-6)

        dXw_np_torch = np.max(np.abs(Xworld_np - Xworld_torch_np))
        dYw_np_torch = np.max(np.abs(Yworld_np - Yworld_torch_np))
        dZw_np_torch = np.max(np.abs(Zworld_np - Zworld_torch_np))
        print(f"Cam2world numpy-vs-torch max abs diff: dX={dXw_np_torch}, dY={dYw_np_torch}, dZ={dZw_np_torch}")
        assert np.allclose(Xworld_np, Xworld_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Yworld_np, Yworld_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Zworld_np, Zworld_torch_np, rtol=1e-6, atol=1e-6)

        dXc_rt_np_torch = np.max(np.abs(Xcam_np_rt - Xcam_torch_rt_np))
        dYc_rt_np_torch = np.max(np.abs(Ycam_np_rt - Ycam_torch_rt_np))
        dZc_rt_np_torch = np.max(np.abs(Zcam_np_rt - Zcam_torch_rt_np))
        print(f"World2cam numpy-vs-torch max abs diff: dX={dXc_rt_np_torch}, dY={dYc_rt_np_torch}, dZ={dZc_rt_np_torch}")
        assert np.allclose(Xcam_np_rt, Xcam_torch_rt_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Ycam_np_rt, Ycam_torch_rt_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Zcam_np_rt, Zcam_torch_rt_np, rtol=1e-6, atol=1e-6)

        du_rt_np_torch = np.max(np.abs(u_np_rt - u_torch_rt_np))
        dv_rt_np_torch = np.max(np.abs(v_np_rt - v_torch_rt_np))
        dd_rt_np_torch = np.max(np.abs(depth_np_rt - depth_torch_rt_np))
        print(f"Cam2img numpy-vs-torch max abs diff: du={du_rt_np_torch}, dv={dv_rt_np_torch}, ddepth={dd_rt_np_torch}")
        assert np.allclose(u_np_rt, u_torch_rt_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(v_np_rt, v_torch_rt_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(depth_np_rt, depth_torch_rt_np, rtol=1e-6, atol=1e-6)

        Xworld_seed_torch = torch.tensor([1.0, 2.0], device=device, dtype=torch.float64)
        Yworld_seed_torch = torch.tensor([0.5, -1.0], device=device, dtype=torch.float64)
        Zworld_seed_torch = torch.tensor([6.0, 8.0], device=device, dtype=torch.float64)

        Xcam_from_world_torch, Ycam_from_world_torch, Zcam_from_world_torch = ori.apply_ext(
            Xworld_seed_torch, Yworld_seed_torch, Zworld_seed_torch, direction="world2cam"
        )
        u_from_world_torch, v_from_world_torch, depth_from_world_torch = ori.apply_K(
            Xcam_from_world_torch, Ycam_from_world_torch, Zcam_from_world_torch, direction="cam2img"
        )
        Xcam_from_img_torch, Ycam_from_img_torch, Zcam_from_img_torch = ori.apply_K(
            u_from_world_torch, v_from_world_torch, depth_from_world_torch, direction="img2cam"
        )
        Xworld_back_torch, Yworld_back_torch, Zworld_back_torch = ori.apply_ext(
            Xcam_from_img_torch, Ycam_from_img_torch, Zcam_from_img_torch, direction="cam2world"
        )

        dXw_torch = torch.max(torch.abs(Xworld_back_torch - Xworld_seed_torch)).item()
        dYw_torch = torch.max(torch.abs(Yworld_back_torch - Yworld_seed_torch)).item()
        dZw_torch = torch.max(torch.abs(Zworld_back_torch - Zworld_seed_torch)).item()
        print(f"Torch CUDA round-trip B max abs diff: dXworld={dXw_torch}, dYworld={dYw_torch}, dZworld={dZw_torch}")

        assert torch.allclose(Xworld_back_torch, Xworld_seed_torch, rtol=1e-6, atol=1e-6)
        assert torch.allclose(Yworld_back_torch, Yworld_seed_torch, rtol=1e-6, atol=1e-6)
        assert torch.allclose(Zworld_back_torch, Zworld_seed_torch, rtol=1e-6, atol=1e-6)

        Xcam_from_world_torch_np = Xcam_from_world_torch.detach().cpu().numpy()
        Ycam_from_world_torch_np = Ycam_from_world_torch.detach().cpu().numpy()
        Zcam_from_world_torch_np = Zcam_from_world_torch.detach().cpu().numpy()
        u_from_world_torch_np = u_from_world_torch.detach().cpu().numpy()
        v_from_world_torch_np = v_from_world_torch.detach().cpu().numpy()
        depth_from_world_torch_np = depth_from_world_torch.detach().cpu().numpy()
        Xcam_from_img_torch_np = Xcam_from_img_torch.detach().cpu().numpy()
        Ycam_from_img_torch_np = Ycam_from_img_torch.detach().cpu().numpy()
        Zcam_from_img_torch_np = Zcam_from_img_torch.detach().cpu().numpy()
        Xworld_back_torch_np = Xworld_back_torch.detach().cpu().numpy()
        Yworld_back_torch_np = Yworld_back_torch.detach().cpu().numpy()
        Zworld_back_torch_np = Zworld_back_torch.detach().cpu().numpy()

        print(
            "Round-trip B numpy-vs-torch max abs diff: "
            f"world2cam dX={np.max(np.abs(Xcam_from_world_np - Xcam_from_world_torch_np))}, "
            f"dY={np.max(np.abs(Ycam_from_world_np - Ycam_from_world_torch_np))}, "
            f"dZ={np.max(np.abs(Zcam_from_world_np - Zcam_from_world_torch_np))}; "
            f"cam2img du={np.max(np.abs(u_from_world_np - u_from_world_torch_np))}, "
            f"dv={np.max(np.abs(v_from_world_np - v_from_world_torch_np))}, "
            f"ddepth={np.max(np.abs(depth_from_world_np - depth_from_world_torch_np))}; "
            f"img2cam dX={np.max(np.abs(Xcam_from_img_np - Xcam_from_img_torch_np))}, "
            f"dY={np.max(np.abs(Ycam_from_img_np - Ycam_from_img_torch_np))}, "
            f"dZ={np.max(np.abs(Zcam_from_img_np - Zcam_from_img_torch_np))}; "
            f"cam2world dX={np.max(np.abs(Xworld_back_np - Xworld_back_torch_np))}, "
            f"dY={np.max(np.abs(Yworld_back_np - Yworld_back_torch_np))}, "
            f"dZ={np.max(np.abs(Zworld_back_np - Zworld_back_torch_np))}"
        )

        assert np.allclose(Xcam_from_world_np, Xcam_from_world_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Ycam_from_world_np, Ycam_from_world_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Zcam_from_world_np, Zcam_from_world_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(u_from_world_np, u_from_world_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(v_from_world_np, v_from_world_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(depth_from_world_np, depth_from_world_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Xcam_from_img_np, Xcam_from_img_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Ycam_from_img_np, Ycam_from_img_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Zcam_from_img_np, Zcam_from_img_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Xworld_back_np, Xworld_back_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Yworld_back_np, Yworld_back_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Zworld_back_np, Zworld_back_torch_np, rtol=1e-6, atol=1e-6)

        assert torch.allclose(u_torch_rt, u_torch, rtol=1e-6, atol=1e-6)
        assert torch.allclose(v_torch_rt, v_torch, rtol=1e-6, atol=1e-6)
        assert torch.allclose(depth_torch_rt, depth_torch, rtol=1e-6, atol=1e-6)
    else:
        print("CUDA not available: skipping torch CUDA round-trip test.")

    # Grid-shape tests (H, W) to validate image-wide projection/backprojection paths.
    Ht, Wt = 4, 5
    uu_np, vv_np = np.meshgrid(np.arange(Wt, dtype=np.float64), np.arange(Ht, dtype=np.float64), indexing='xy')
    depth_grid_np = 2.0 + 0.01 * uu_np + 0.02 * vv_np

    Xg_np, Yg_np, Zg_np = ori.apply_K(uu_np, vv_np, depth_grid_np, direction="img2cam")
    assert Xg_np.shape == (Ht, Wt)
    assert Yg_np.shape == (Ht, Wt)
    assert Zg_np.shape == (Ht, Wt)
    assert np.isfinite(Xg_np).all() and np.isfinite(Yg_np).all() and np.isfinite(Zg_np).all()

    uu_np_rt, vv_np_rt, dd_np_rt = ori.apply_K(Xg_np, Yg_np, Zg_np, direction="cam2img")
    assert np.allclose(uu_np_rt, uu_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(vv_np_rt, vv_np, rtol=1e-5, atol=1e-5)
    assert np.allclose(dd_np_rt, depth_grid_np, rtol=1e-5, atol=1e-5)

    Xw_grid_np, Yw_grid_np, Zw_grid_np = ori.apply_ext(Xg_np, Yg_np, Zg_np, direction="cam2world")
    Xc_grid_np_rt, Yc_grid_np_rt, Zc_grid_np_rt = ori.apply_ext(
        Xw_grid_np, Yw_grid_np, Zw_grid_np, direction="world2cam"
    )
    assert np.allclose(Xc_grid_np_rt, Xg_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(Yc_grid_np_rt, Yg_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(Zc_grid_np_rt, Zg_np, rtol=1e-6, atol=1e-6)

    if torch.cuda.is_available():
        uu_t = torch.from_numpy(uu_np).to(device=device, dtype=torch.float64)
        vv_t = torch.from_numpy(vv_np).to(device=device, dtype=torch.float64)
        dd_t = torch.from_numpy(depth_grid_np).to(device=device, dtype=torch.float64)

        Xg_t, Yg_t, Zg_t = ori.apply_K(uu_t, vv_t, dd_t, direction="img2cam")
        assert Xg_t.shape == (Ht, Wt)
        assert Yg_t.shape == (Ht, Wt)
        assert Zg_t.shape == (Ht, Wt)
        assert torch.isfinite(Xg_t).all() and torch.isfinite(Yg_t).all() and torch.isfinite(Zg_t).all()

        uu_t_rt, vv_t_rt, dd_t_rt = ori.apply_K(Xg_t, Yg_t, Zg_t, direction="cam2img")
        assert torch.allclose(uu_t_rt, uu_t, rtol=1e-6, atol=1e-6)
        assert torch.allclose(vv_t_rt, vv_t, rtol=1e-6, atol=1e-6)
        assert torch.allclose(dd_t_rt, dd_t, rtol=1e-6, atol=1e-6)

        Xw_t, Yw_t, Zw_t = ori.apply_ext(Xg_t, Yg_t, Zg_t, direction="cam2world")
        Xc_t_rt, Yc_t_rt, Zc_t_rt = ori.apply_ext(Xw_t, Yw_t, Zw_t, direction="world2cam")
        assert torch.allclose(Xc_t_rt, Xg_t, rtol=1e-6, atol=1e-6)
        assert torch.allclose(Yc_t_rt, Yg_t, rtol=1e-6, atol=1e-6)
        assert torch.allclose(Zc_t_rt, Zg_t, rtol=1e-6, atol=1e-6)

        Xk_t, Yk_t, Zk_t = ori.apply_K_torch(depth_grid_np, device=device)
        assert Xk_t.shape == (Ht, Wt)
        assert Yk_t.shape == (Ht, Wt)
        assert Zk_t.shape == (Ht, Wt)
    else:
        Xk_cpu, Yk_cpu, Zk_cpu = ori.apply_K_torch(depth_grid_np, device="cpu")
        assert Xk_cpu.shape == (Ht, Wt)
        assert Yk_cpu.shape == (Ht, Wt)
        assert Zk_cpu.shape == (Ht, Wt)

    print("Grid-shape K/ext tests passed.")

    print("All K/ext round-trip and numpy-vs-torch tests passed.")
    

    

