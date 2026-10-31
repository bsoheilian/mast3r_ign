# create the sckeleton for aerial and ground data loader by planning
# it should inheritate from BaseStereoViewDataset like both MegaDepthAerial and MASt3RBaseStereoViewDataset
# it should implement dataloader and __getitem__
# it should manage data augmentation 
# it should handle resolution and required crops correctly (not like MASt3RBaseStereoViewDataset)
# it should provide a example at the end to load randomly data and visualize it 
# it should be usable by ign_aerial_ground_train (see /home/BSoheilian/work/dev/mast3r_ign/demo_duster_viz.py for visualization example)



"""
    design: 
        - The loader reads for intialization a metadata that contains minimum required information
        to understand pairs to which a site belongs (the site is used to seperate learning and test datasets)
        - __getitem__ should return views (that is a pair of images with all required associated info)
        - a random data augmentation is always there and controlled by:
            - aug_crop, aug_swap, aug_portrait_or_landscape, aug_rot90
        - a random number generator is used to control the augmentations
        - understand required member of base class:
            - split is either 'train', 'val', or 'test' depending on the dataset usage
            it defines then rules to seperate the data into training, validation, and test sets depending most of the time
            on scene  id 

            - transform:
            transform is preprocessing applied to each image in __getitem__, 
            after cropping. The default ImgNorm converts the image to a channel-first
            tensor and normalizes pixel values from approximately [0, 1] to [-1, 1]. 
            It applies to the image, not the depth map or camera parameters. 
            A caller can pass a different transform; for example, training configurations 
            may use ColorJitter followed by ImgNorm

    usage: 
    the train script use the data loadr as follows : 
    torchrun --nproc_per_node=8 train.py \
    --train_dataset "57_000 @ Habitat512(1_000_000, split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 68_400 @ BlendedMVS(split='train', mask_sky=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 68_400 @ MegaDepth(split='train', mask_sky=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ ARKitScenes(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ Co3d(split='train', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ StaticThings3D(mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ ScanNetpp(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 45_600 @ TartanAir(pairs_subset='', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 4_560 @ UnrealStereo4K(resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 1_140 @ VirtualKitti(optical_center_is_centered=True, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 22_800 @ WildRgbd(split='train', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 145_920 @ NianticMapFree(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 57_000 @ DL3DV(split='nlight', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 57_000 @ DL3DV(split='not-nlight', cosxl_augmentations=None, resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5) + 34_200 @ InternalUnreleasedDataset(resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, transform=ColorJitter, n_corres=8192, nneg=0.5)" \
    --test_dataset "Habitat512(1_000, split='val', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ BlendedMVS(split='val', resolution=(512,384), mask_sky=True, seed=777, n_corres=1024) + 1_000 @ ARKitScenes(split='test', resolution=(512,384), seed=777, n_corres=1024) + 1_000 @ MegaDepth(split='val', mask_sky=True, resolution=(512,336), seed=777, n_corres=1024) + 1_000 @ Co3d(split='test', resolution=(512,384), mask_bg='rand', seed=777, n_corres=1024)" \
    
    for training it use 57k samples/epoch from the Habitat512 data loader and 
    68400 samples/epoch from BlendedMVS data loader

    It seems that if aug_crop_arg = False then the cropping does not happen and so the 
    code may become usable as it does not crop the already cropped images.



"""

import numpy as np
import PIL.Image
import torch

import mast3r.utils.path_to_dust3r
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset, is_good_type
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates


class ign_aerial_ground(BaseStereoViewDataset):
    """IGN pair dataset with custom I/O and no implicit geometry augmentation.

    Implement _load_pairs and _get_views for the IGN storage format. Loaded views
    must already match the single requested (width, height) resolution.
    """

    def __init__(self, *, ROOT, resolution, split=None, transform=ImgNorm,
                 seed=None, is_metric_scale=False, n_corres=0):
        super().__init__(split=split, resolution=resolution, transform=transform, seed=seed)
        assert len(self._resolutions) == 1, 'Only one resolution is supported'
        assert isinstance(n_corres, int) and n_corres >= 0
        self.ROOT = ROOT
        self.is_metric_scale = bool(is_metric_scale)
        self.n_corres = n_corres
        self.pairs = self._load_pairs()

    def _load_pairs(self):
        """Return pair metadata filtered by self.split, keeping sites disjoint."""
        raise NotImplementedError('Implement IGN pair metadata loading in _load_pairs')

    def __len__(self):
        return len(self.pairs)

    def _get_views(self, idx, resolution, rng):
        """Load self.pairs[idx] and return two fresh view dictionaries.

        Required keys: img (RGB PIL image), depthmap (float32 camera-z depth),
        camera_intrinsics (float32 3x3), camera_pose (float32 4x4 camera-to-world),
        dataset, label (site identifier), and instance (image identifier).
        Both cameras must use the same world frame and units. Invalid depth is
        zero; negative depth denotes sky. Do custom augmentation here, keeping
        geometry and correspondences aligned and preserving the target size.

        When n_corres > 0, also return corres (n_corres, 2) integer (x, y)
        coordinates and valid_corres (n_corres,) boolean masks in both views.
        Coordinates must be in bounds even for negative matching samples;
        masks must agree between views. No correspondence generation is imposed.
        """
        raise NotImplementedError('Implement IGN image, depth and camera loading in _get_views')

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, ar_idx = idx
        else:
            ar_idx = 0
        assert ar_idx == 0, 'Only resolution index 0 is supported'
        if not 0 <= idx < len(self):
            raise IndexError(idx)

        if self.seed is not None:
            self._rng = np.random.default_rng(seed=self.seed + int(idx))
        elif not hasattr(self, '_rng'):
            self._rng = np.random.default_rng(seed=torch.initial_seed())

        resolution = self._resolutions[ar_idx]
        width, height = resolution
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views, 'Expected exactly two views'

        for view_idx, view in enumerate(views):
            assert isinstance(view['img'], PIL.Image.Image) and view['img'].mode == 'RGB'
            assert view['img'].size == resolution, f'Expected image size {resolution}'
            assert view['depthmap'].shape == (height, width), 'Depth resolution mismatch'
            assert view['camera_intrinsics'].shape == (3, 3)
            assert view['camera_pose'].shape == (4, 4)
            for key in ('depthmap', 'camera_intrinsics', 'camera_pose'):
                assert view[key].dtype == np.float32, f'{key} must be float32'
                assert np.isfinite(view[key]).all(), f'Non-finite {key}'
            for key in ('dataset', 'label', 'instance'):
                assert isinstance(view[key], str), f'{key} must be a string'

            view['idx'] = (int(idx), int(ar_idx), view_idx)
            view['true_shape'] = np.int32((height, width))
            view['is_metric_scale'] = self.is_metric_scale
            view['sky_mask'] = view['depthmap'] < 0
            view['img'] = self.transform(view['img'])
            assert isinstance(view['img'], torch.Tensor)
            assert view['img'].shape == (3, height, width), 'Transform changed image dimensions'
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            if self.n_corres:
                assert 'corres' in view and 'valid_corres' in view, (
                    '_get_views must provide matching targets when n_corres > 0')
                corres = view['corres']
                assert corres.shape == (self.n_corres, 2)
                assert np.issubdtype(corres.dtype, np.integer), 'Correspondences must index pixels'
                assert ((corres >= 0) & (corres < (width, height))).all()
                assert view['valid_corres'].shape == (self.n_corres,)
                assert view['valid_corres'].dtype == np.bool_

            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
            for key, value in view.items():
                good_type, message = is_good_type(key, value)
                assert good_type, f'{key}: {message}'

        if self.n_corres:
            assert np.array_equal(views[0]['valid_corres'], views[1]['valid_corres'])
        return views


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize an IGN pair after implementing the storage hooks')
    parser.add_argument('--root', required=True)
    parser.add_argument('--split', default='train', choices=('train', 'val', 'test'))
    parser.add_argument('--resolution', nargs=2, type=int, default=(512, 384), metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    dataset = ign_aerial_ground(ROOT=args.root, split=args.split,
                                resolution=tuple(args.resolution), seed=args.seed)
    if not len(dataset):
        parser.error('No pairs in the selected split')
    sample_idx = int(np.random.default_rng(args.seed).integers(len(dataset)))
    views = dataset[sample_idx]

    from dust3r.utils.image import rgb
    from dust3r.viz import SceneViz, auto_cam_size

    viz = SceneViz()
    camera_size = auto_cam_size(np.stack([view['camera_pose'] for view in views]))
    for view, color in zip(views, ((255, 0, 0), (0, 255, 0))):
        image = rgb(view['img'])
        viz.add_pointcloud(view['pts3d'], color=image, mask=view['valid_mask'])
        viz.add_camera(pose_c2w=view['camera_pose'],
                       focal=view['camera_intrinsics'][0, 0], color=color,
                       image=image, cam_size=camera_size)
    viz.show()

