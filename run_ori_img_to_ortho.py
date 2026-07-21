import argparse
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from oriented_image_to_ortho.ortho_from_oriented_img import oriented_img_to_ortho



def _load_st_orientation():
    """Import STOrientation and undistort_tif from dchan with a small fallback path setup."""
    try:
        from dchan.core.utils.stereopolis_orientation import STOrientation, undistort_tif
        return STOrientation, undistort_tif
    except ModuleNotFoundError:
        candidates = [
            Path(__file__).resolve().parent.parent / "dchan",
            Path("/dchan"),
        ]
        for candidate in candidates:
            if candidate.exists():
                sys.path.insert(0, str(candidate.resolve()))
                break
        from dchan.core.utils.stereopolis_orientation import STOrientation, undistort_tif
        return STOrientation, undistort_tif

def get_args_parser():
    parser = argparse.ArgumentParser('my_tool', add_help=True)

    # Model args
    model_path = './docker/files/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    parser.add_argument('--weights',        default=model_path, type=str, help='path to model weights file')
    parser.add_argument('--rgb_img',        default='./ign_samples/rgb_sample.jpg', type=str, help='path to the RGB image file')
    parser.add_argument('--rot',            default=None, type=str, help='path to the rotation file (use with --trans)')
    parser.add_argument('--trans',          default=None, type=str, help='path to the translation file (use with --rot)')
    parser.add_argument('--ori',            default=None, type=str, help='path to orientation XML file (alternative to --rot/--trans)')
    parser.add_argument('--z_scale',        default=1.0, type=float, help='z scale factor')
    parser.add_argument('--gsd',            default=0.1, type=float, help='ground sample distance')
    parser.add_argument('--out_dir',        default='./ign_samples/output', type=str, help='path to the output directory')
    parser.add_argument('--debug_tmp_copy_dir', default=None, type=str, help='if set, copy temp rgb/rotation/translation files here for debugging')
    

    return parser


def _copy_debug_inputs(debug_dir, rgb_img_file, rotation_file, translation_file):
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    copied = []
    for src_path, label in [
        (rgb_img_file, 'rgb_img_file'),
        (rotation_file, 'rotation_file'),
        (translation_file, 'translation_file'),
    ]:
        src = Path(src_path)
        if src.exists():
            dst = debug_path / f'{label}_{stamp}{src.suffix}'
            shutil.copy2(src, dst)
            copied.append(str(dst))

    if copied:
        print('Debug copies written:')
        for item in copied:
            print(f'  - {item}')


def _validate_args(parser, args):
    has_ori = args.ori is not None
    has_rot = args.rot is not None
    has_trans = args.trans is not None

    if has_ori and (has_rot or has_trans):
        parser.error('Use either --ori OR both --rot and --trans, not both modes at once.')
    if not has_ori and not (has_rot and has_trans):
        parser.error('Provide --ori, or provide both --rot and --trans.')

if __name__ == '__main__':
    from mast3r.utils.host_path import container_to_host

    parser = get_args_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    rotation_file = args.rot
    translation_file = args.trans
    rgb_img_file = args.rgb_img

    with tempfile.TemporaryDirectory(prefix='ori_to_rt_') as tmpdir:
        if args.ori is not None:
            STOrientation, undistort_tif = _load_st_orientation()
            ori = STOrientation(args.ori)

            rotation_file = str(Path(tmpdir) / 'R.txt')
            translation_file = str(Path(tmpdir) / 'T.txt')
            np.savetxt(rotation_file, ori.extrinsic.rotation_matrix)
            np.savetxt(translation_file, ori.extrinsic.position)
            # to write undistort image here 
            rgb_img_file = str(Path(tmpdir) / 'undistorted.jpg')
            undistort_tif((args.rgb_img), rgb_img_file, ori.distortion_polynomial)

        if args.debug_tmp_copy_dir:
            _copy_debug_inputs(
                args.debug_tmp_copy_dir,
                rgb_img_file,
                rotation_file,
                translation_file,
            )

            
            

        oriented_img_to_ortho(
            args.weights,
            rgb_img_file,
            rotation_file,
            translation_file,
            z_scale=args.z_scale,
            gsd=args.gsd,
            output_dir=args.out_dir,
            str_output_dir_in_host=container_to_host(args.out_dir),
        )
