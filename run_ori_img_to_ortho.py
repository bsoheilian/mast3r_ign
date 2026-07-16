# your_launcher.py
import argparse

from oriented_image_to_ortho.ortho_from_oriented_img import oriented_img_to_ortho

def get_args_parser():
    parser = argparse.ArgumentParser('my_tool', add_help=True)

    # Model args
    model_path = './docker/files/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    parser.add_argument('--weights',        default=model_path, type=str, help='path to model weights file')
    parser.add_argument('--rgb_img',        default='./ign_samples/rgb_sample.jpg', type=str, help='path to the RGB image file')
    parser.add_argument('--rot',            default='./ign_samples/R.txt', type=str, help='path to the rotation file')
    parser.add_argument('--trans',          default='./ign_samples/T.txt', type=str, help='path to the translation file')
    parser.add_argument('--z_scale',        default=1.0, type=float, help='z scale factor')
    parser.add_argument('--gsd',            default=0.1, type=float, help='ground sample distance')
    parser.add_argument('--out_dir',        default='./ign_samples/output', type=str, help='path to the output directory')
    parser.add_argument('--out_dir_host',   default=None, type=str, help='path to the output directory')

    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    oriented_img_to_ortho(args.weights, args.rgb_img, args.rot, args.trans, z_scale=args.z_scale, gsd=args.gsd,
                          output_dir=args.out_dir,
                          str_output_dir_in_host=args.out_dir_host)
