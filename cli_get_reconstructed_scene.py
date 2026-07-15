#!/usr/bin/env python3
import sys
import os
import numpy as np
from mast3r.model import AsymmetricMASt3R
sys.path.insert(0, '/home/BSoheilian/work/dev/mast3r_ign')

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
    

def main():
    """
    CLI interface to call get_reconstructed_scene with initialized parameters
    """
    # Initialize all parameters
    outdir = './output'
    gradio_delete_cache = False
    weights_path = './docker/files/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    retrieval_model = None
    device = 'cuda'  # or 'cuda' if available
    silent = False
    image_size = 512
    current_scene_state = None
    # filelist = ['./data/Lille-150127_0485-11-00002_0000384.jpg']  # List of image file paths
    #todo Bahman how to handle the images rotated 180 deg around Z that causes serieus anomalies when AI images are applied on. 
    filelist = ['./data/chantier_lilles/pvt/Paris-140613_0494-301-00003_0000191_undist_rot.jpg']  # List of image file paths
    optim_level = 'refine' # it does not make sense to use refine+depth as the input is a single image only refine is ok 
    lr1 = 0.07
    niter1 = 300
    lr2 = 0.01
    niter2 = 300
    min_conf_thr = 1.5
    matching_conf_thr = 0.0
    as_pointcloud = True
    mask_sky = False
    clean_depth = False
    transparent_cams = False
    cam_size = 0.2
    scenegraph_type = 'complete'
    winsize = 1
    win_cyclic = False
    refid = 0
    TSDF_thresh = 0.0
    shared_intrinsics = False  #setting to true shall use the same intrinsics but strangly introduces big error when only one image is used.
    subsample = 1
    
    # Call get_reconstructed_scene
    model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')
    img_rgb,img_depth,intrinsics,extrinsics = get_reconstructed_scene(
        outdir=outdir,
        gradio_delete_cache=gradio_delete_cache,
        model=model,
        retrieval_model=retrieval_model,
        device=device,
        silent=silent,
        image_size=image_size,
        current_scene_state=current_scene_state,
        filelist=filelist,
        optim_level=optim_level,
        lr1=lr1,
        niter1=niter1,
        lr2=lr2,
        niter2=niter2,
        min_conf_thr=min_conf_thr,
        matching_conf_thr=matching_conf_thr,
        as_pointcloud=as_pointcloud,
        mask_sky=mask_sky,
        clean_depth=clean_depth,
        transparent_cams=transparent_cams,
        cam_size=cam_size,
        scenegraph_type=scenegraph_type,
        winsize=winsize,
        win_cyclic=win_cyclic,
        refid=refid,
        TSDF_thresh=TSDF_thresh,
        shared_intrinsics=shared_intrinsics,
        subsample=subsample,
        cli_call=True
    )
    cli_write_results_to_files(img_rgb,img_depth,intrinsics,extrinsics, outdir)



    


if __name__ == '__main__':
    main()

