
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/mast3r_ign')
from oriented_image_to_ortho.depth_from_img import infer_depth_from_img
from oriented_image_to_ortho.ortho_from_dept_and_ori import dept_and_ori_to_ortho


def oriented_img_to_ortho(weights_path, img_rgb, rotation_file, translation_file, z_scale=1.0, gsd=0.05, 
                          output_dir="./output", str_output_dir_in_host=None ):
    """
    Convert an oriented image to an orthographic image using depth and orientation information.

    Parameters:
    - img_rgb: RGB image as a numpy array.
    - weights_path: Path to the weights for depth inference.
    - img_rgb: Path to the RGB image file.
    - rotation: Camera rotation matrix.
    - translation: Camera translation vector.
    - z_scale: Scale factor for depth values (default is 1.0).
    - gsd: Ground sample distance for the orthographic image (default is 0.05).
    - output_dir: Directory to save the output files (default is "./output").
    - str_output_dir_in_host: Optional string for the output directory in the host system.

    Returns:
    None
    """
    import os
    img_name_without_ext = os.path.splitext(os.path.basename(img_rgb))[0]
    output_ortho_img_filename = f"ortho_{img_name_without_ext}_z_scale_{z_scale}_gsd_{gsd}"

    img_rgb_py, img_depth, intrinsics,_= infer_depth_from_img(weights_path, imgfilename=img_rgb)
    

    rotation = np.loadtxt(rotation_file, dtype=np.float64)
    translation = np.loadtxt(translation_file, dtype=np.float32)

    dept_and_ori_to_ortho(img_rgb_py, img_depth, intrinsics, rotation, translation, z_scale=z_scale, gsd=gsd,
                          output_dir=output_dir, output_ortho_img_filename=output_ortho_img_filename,
                          str_output_dir_in_host=str_output_dir_in_host)
    
if __name__ == "__main__":
    weights_path = './docker/files/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    img_rgb = './ign_samples/rgb_sample.jpg'
    rotation = './ign_samples/R.txt'
    translation ='./ign_samples/T.txt'
    
    oriented_img_to_ortho(weights_path, img_rgb, rotation, translation, z_scale=2.5, gsd=0.05,
                          output_dir="./ign_samples/output/",
                          str_output_dir_in_host="/home/BSoheilian/work/dev/mast3r_ign/ign_samples/output/")
