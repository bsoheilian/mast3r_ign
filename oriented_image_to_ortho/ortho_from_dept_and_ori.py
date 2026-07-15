
import sys
from pathlib import Path

from matplotlib import pyplot as plt


sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from renderer.mesh_from_3D import TexturedMesh3D
from ori_img_utils.ori import Orientation
from renderer.export_to_ply import export_ply, export_xyzrgb_points_to_ply


def dept_and_ori_to_ortho(img_rgb, img_depth, intrinsics, rotation, translation, z_scale=1.0, gsd=0.05, 
                          output_dir="./output", output_ortho_img_filename="ortho", str_output_dir_in_host=None ):

    ori = Orientation.from_arrays(intrinsics, rotation, translation)

    Xcam, Ycam, Zcam = ori.apply_K_torch(img_depth, z_scale=z_scale, device="cuda")
    Xg, Yg, Zg = ori.apply_ext(Xcam, Ycam, Zcam, direction="cam2world")
    # export_xyzrgb_points_to_ply(Xg, Yg, Zg, 255*img_rgb, path="./output/points_world_PR.ply")
    mesh = TexturedMesh3D(Xg, Yg, Zg, img_rgb)
    # export_ply(mesh.mesh, path="./output/refactor_mesh.ply") 

    img = mesh.create_orth(
        gsd=gsd,
        profile=True,
        safe_raster=True,
        cull_backfaces=True,
        use_hard_shader=True,
        faces_per_pixel = 1,
        blur_radius = 1e-30,
        verbose=True
    )
    ortho_img_full_path = Path(output_dir) / f"{output_ortho_img_filename}.png"
    plt.imsave(ortho_img_full_path, img.cpu().numpy())
    ortho_img_full_path_in_host = Path(str_output_dir_in_host) / f"{output_ortho_img_filename}.png" if str_output_dir_in_host is not None else ortho_img_full_path
    mesh._write_vrt_qgis_safe(
        vert_file_path=Path(output_dir) / f"{output_ortho_img_filename}.vrt",
        ortho_img_path=ortho_img_full_path_in_host,
            gsd=gsd,
            ortho_img_size=(img.shape[0], img.shape[1])
        )



if __name__ == "__main__":
    K = np.loadtxt('./ign_samples/output/intrinsic.txt', dtype=np.float32)
    R = np.loadtxt('./ign_samples/R.txt', dtype=np.float64)
    C = np.loadtxt('./ign_samples/T.txt', dtype=np.float32)
    print(f"Intrinsic matrix K:\n{K}\nRotation matrix R:\n{R}\nTranslation vector C:\n{C}")

    img_rgb = np.load('./ign_samples/output/rgb_image.npy')
    H, W, _ = img_rgb.shape
    img_depth = np.load('./ign_samples/output/depth_image.npy')
    img_depth = img_depth.reshape(H, -1)  # Reshape to (H, W)
    print(f"RGB image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}, min: {img_rgb.min()}, max: {img_rgb.max()}, type: {type(img_rgb)}")
    print(f"Depth image shape: {img_depth.shape}, dtype: {img_depth.dtype}, type: {type(img_depth)}, min: {img_depth.min()}, max: {img_depth.max()}")

    dept_and_ori_to_ortho(img_rgb, img_depth, K, R, C, z_scale=2.5, gsd=0.05, 
                          output_dir="./ign_samples/output/", 
                          output_ortho_img_filename="ortho", 
                          str_output_dir_in_host="/home/BSoheilian/work/dev/mast3r_ign/ign_samples/output/")
