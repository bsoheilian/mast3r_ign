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
        
        return cls(K, R, C)
    
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
            x = torch.atleast_1d(torch.as_tensor(x))
            y = torch.atleast_1d(torch.as_tensor(y))
            z = torch.atleast_1d(torch.as_tensor(z))
            K = torch.from_numpy(self._K).to(x.dtype).to(x.device)

            if direction == "img2cam":
                pixels = torch.stack([x, y, torch.ones_like(x)])
                normalized = torch.linalg.inv(K) @ pixels
                return (normalized[0] * z, normalized[1] * z, normalized[2] * z)
            points = torch.stack([x, y, z])
            projected = K @ points
            return (projected[0] / projected[2], projected[1] / projected[2], projected[2])

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        if direction == "img2cam":
            pixels = np.array([x, y, np.ones_like(x)])
            normalized = np.linalg.inv(self._K) @ pixels
            return (normalized[0] * z, normalized[1] * z, normalized[2] * z)
        points = np.array([x, y, z])
        projected = self._K @ points
        return (projected[0] / projected[2], projected[1] / projected[2], projected[2])


if __name__ == "__main__":
    # Create test matrices
    print("Testing Orientation class...")
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    C = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # Initialize Orientation with ndarrays
    ori = Orientation.from_arrays(K, R, C)
    print("Orientation initialized successfully")
    
    # Round-trip test with numpy inputs: img -> cam -> img
    u_np = np.array([320.0, 640.0])
    v_np = np.array([240.0, 480.0])
    depth_np = np.array([2.0, 3.0])
    
    Xcam_np, Ycam_np, Zcam_np = ori.apply_K(u_np, v_np, depth_np, direction="img2cam")
    u_np_rt, v_np_rt, depth_np_rt = ori.apply_K(Xcam_np, Ycam_np, Zcam_np, direction="cam2img")

    du_np = np.max(np.abs(u_np_rt - u_np))
    dv_np = np.max(np.abs(v_np_rt - v_np))
    dd_np = np.max(np.abs(depth_np_rt - depth_np))
    print(f"Numpy max abs diff: du={du_np}, dv={dv_np}, ddepth={dd_np}")
    assert np.allclose(u_np_rt, u_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(v_np_rt, v_np, rtol=1e-6, atol=1e-6)
    assert np.allclose(depth_np_rt, depth_np, rtol=1e-6, atol=1e-6)
    
    # Round-trip test with torch CUDA inputs: img -> cam -> img
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Use float64 to make strict 1e-6 round-trip checks numerically stable on CUDA.
        u_torch = torch.tensor([320.0, 640.0], device=device, dtype=torch.float64)
        v_torch = torch.tensor([240.0, 480.0], device=device, dtype=torch.float64)
        depth_torch = torch.tensor([2.0, 3.0], device=device, dtype=torch.float64)

        Xcam_torch, Ycam_torch, Zcam_torch = ori.apply_K(u_torch, v_torch, depth_torch, direction="img2cam")
        u_torch_rt, v_torch_rt, depth_torch_rt = ori.apply_K(Xcam_torch, Ycam_torch, Zcam_torch, direction="cam2img")

        du_torch = torch.max(torch.abs(u_torch_rt - u_torch)).item()
        dv_torch = torch.max(torch.abs(v_torch_rt - v_torch)).item()
        dd_torch = torch.max(torch.abs(depth_torch_rt - depth_torch)).item()
        print(f"Torch CUDA max abs diff: du={du_torch}, dv={dv_torch}, ddepth={dd_torch}")

        Xcam_torch_np = Xcam_torch.detach().cpu().numpy()
        Ycam_torch_np = Ycam_torch.detach().cpu().numpy()
        Zcam_torch_np = Zcam_torch.detach().cpu().numpy()
        dX_np_torch = np.max(np.abs(Xcam_np - Xcam_torch_np))
        dY_np_torch = np.max(np.abs(Ycam_np - Ycam_torch_np))
        dZ_np_torch = np.max(np.abs(Zcam_np - Zcam_torch_np))
        print(f"Img2cam numpy-vs-torch max abs diff: dX={dX_np_torch}, dY={dY_np_torch}, dZ={dZ_np_torch}")
        assert np.allclose(Xcam_np, Xcam_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Ycam_np, Ycam_torch_np, rtol=1e-6, atol=1e-6)
        assert np.allclose(Zcam_np, Zcam_torch_np, rtol=1e-6, atol=1e-6)

        assert torch.allclose(u_torch_rt, u_torch, rtol=1e-6, atol=1e-6)
        assert torch.allclose(v_torch_rt, v_torch, rtol=1e-6, atol=1e-6)
        assert torch.allclose(depth_torch_rt, depth_torch, rtol=1e-6, atol=1e-6)
    else:
        print("CUDA not available: skipping torch CUDA round-trip test.")

    print("Round-trip tests passed for numpy and torch.")
    

    

