#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '/mast3r_ign')

import numpy as np
import matplotlib.pyplot as plt
from mast3r.model import AsymmetricMASt3R
from mast3r.demo import get_reconstructed_scene

def cli_write_results_to_files(img_rgb,img_depth,intrinsics,extrinsics, outdir):
 


    img_rgb_file = os.path.join(outdir, 'rgb_image.npy')
    img_depth_file = os.path.join(outdir, 'depth_image.npy')
    intrinsics_file = os.path.join(outdir, 'intrinsic.txt')
    extrinsics_file = os.path.join(outdir, 'pose.txt')
        
    np.save(img_rgb_file, img_rgb )
    np.save(img_depth_file, img_depth)
    np.savetxt(intrinsics_file, intrinsics)
    np.savetxt(extrinsics_file, extrinsics)
    

    print(f"Saved RGB image of size {img_rgb.shape} to: {img_rgb_file}")
    print(f"Saved depth map of size {img_depth.shape} to: {img_depth_file}")
    print(f"Saved intrinsic matrix {intrinsics} to: {intrinsics_file}")
    print(f"Saved camera pose {extrinsics} to: {extrinsics_file}")
    

def infer_depth_from_img(weights_path = None, imgfilename=None):
                         
    """
    CLI interface to call get_reconstructed_scene with initialized parameters
    """
  
    model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')
    return get_reconstructed_scene(outdir='',
                                   gradio_delete_cache=False,
                                   model=model,
                                   retrieval_model = None,
                                   device='cuda',
                                   silent=False,
                                   image_size=512,
                                   current_scene_state=None,
                                   filelist=[imgfilename],
                                   optim_level='refine',
                                   lr1 = 0.07,
                                   niter1 = 300,
                                   lr2 = 0.01,
                                   niter2 = 300,
                                   min_conf_thr = 1.5,
                                   matching_conf_thr = 0.0,
                                   as_pointcloud = True,
                                   mask_sky = False,
                                   clean_depth = False,
                                   transparent_cams = False,
                                   cam_size = 0.2, 
                                   scenegraph_type = 'complete',
                                   winsize = 1,
                                   win_cyclic = False,
                                   refid = 0,
                                   TSDF_thresh = 0.0,
                                   shared_intrinsics = False,  #setting to true shall use the same intrinsics but strangly introduces big error when only one image is used.
                                   subsample = 1,
                                   cli_call=True)

if __name__ == '__main__':
    weights_path = './docker/files/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    img_rgb,img_depth,intrinsics,extrinsics = infer_depth_from_img(weights_path = weights_path, imgfilename='./ign_samples/rgb_sample.jpg')
    cli_write_results_to_files(img_rgb,img_depth,intrinsics,extrinsics, outdir='./ign_samples/output')
    # plt.figure()
    # plt.imshow(img_depth, cmap='viridis', vmin=0.0, vmax=30.0)
    # plt.colorbar()
    # plt.title('Depth Map')
    # plt.show()
    

