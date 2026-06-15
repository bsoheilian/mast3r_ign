import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, OrthographicCameras
from geometric_operation import OrthoRectifier



def read_depth_image(file_path, width:int):
    data = np.load(file_path)
    depth_image = data.reshape(-1, width)
    return depth_image

def read_rgb_image(file_path):
    data = np.load(file_path)
    return data

def display_images(rgb, depth):
    # Display both images in a single column (top and bottom)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 12), squeeze=False)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(depth, cmap="turbo", vmin=0, vmax=4.0)
    axes[1, 0].set_title('Depth Image')
    axes[1, 0].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    np.set_printoptions(precision=12, suppress=True)
    rgp_path = "./output/debug_rgb_Lille-150127_0485-11-00002_0000384.jpg_2.npy"
    depth_path = "./output/debug_depthmap_Lille-150127_0485-11-00002_0000384.jpg_2.npy"
    parser = argparse.ArgumentParser(
        description='Process depth and RGB images',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--depth-path', type=str, required=False, default=depth_path, help='Path to depth image file')
    parser.add_argument('--rgb-path', type=str, required=False, default=rgp_path, help='Path to RGB image file')
        
    args = parser.parse_args()
    width = 0
    try:
        rgb = read_rgb_image(args.rgb_path)
        width = rgb.shape[1]
        print(f"RGB image {type(rgb)} shape: {rgb.shape}")
    except FileNotFoundError:
        print(f"Warning: RGB image not found at {args.rgb_path}")
       
    try:
        depth = read_depth_image(args.depth_path, width)
        print(f"Depth image {type(depth)} shape: {depth.shape}")
    except FileNotFoundError:
        print(f"Warning: Depth image not found at {args.depth_path}")

    with open('/mast3r_ign/output/intrinsics.txt') as f:
        k = np.array(eval(f.read()))
        print(f"Camera intrinsics:\n{k} {type(k)}")

    with open('/mast3r_ign/output/rot_copy.txt') as f:
        rot = np.array(eval(f.read()))
        print(f"Camera rotation:\n{rot} {type(rot)}")

    with open('/mast3r_ign/output/trans.txt') as f:
        trans = np.array(eval(f.read()))
        print(f"Camera translation:\n{trans} {type(trans)}")
        
    

    ortho_rectifier = OrthoRectifier(rgb, depth, k, rot, trans)
    ortho_rectifier.run()
    
    
if __name__ == "__main__":
    main()