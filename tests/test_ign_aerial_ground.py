import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from misc_ign_data.ign_aerial_ground import ign_aerial_ground
from dust3r.datasets import get_data_loader


class SyntheticIGN(ign_aerial_ground):
    def _load_pairs(self):
        return [0, 1, 2, 3]

    def _get_views(self, idx, resolution, rng):
        width, height = resolution
        views = []
        for view_idx in range(2):
            depth = np.full((height, width), 2, dtype=np.float32)
            depth[0, 0] = 0
            depth[0, 1] = -1
            view = dict(
                img=Image.new('RGB', resolution, (int(rng.integers(256)), 64, 128)),
                depthmap=depth,
                camera_intrinsics=np.eye(3, dtype=np.float32),
                camera_pose=np.eye(4, dtype=np.float32),
                dataset='synthetic', label='site', instance=f'{idx}_{view_idx}',
            )
            if self.n_corres:
                view['corres'] = np.zeros((self.n_corres, 2), dtype=np.int64)
                view['valid_corres'] = np.ones(self.n_corres, dtype=bool)
            views.append(view)
        return views


class TestIgnAerialGround(unittest.TestCase):
    def dataset(self, **kwargs):
        return SyntheticIGN(ROOT='unused', resolution=(8, 4), seed=0, **kwargs)

    def test_geometry_and_seed_zero(self):
        dataset = self.dataset(is_metric_scale=True)
        with patch.object(dataset, '_crop_resize_if_necessary', side_effect=AssertionError('No cropping')):
            views = dataset[0]
        repeated = dataset[0, 0]
        self.assertEqual(len(views), 2)
        for view, again in zip(views, repeated):
            self.assertTrue(torch.equal(view['img'], again['img']))
            self.assertEqual(view['rng'], again['rng'])
            self.assertEqual(tuple(view['img'].shape), (3, 4, 8))
            np.testing.assert_array_equal(view['true_shape'], (4, 8))
            np.testing.assert_allclose(view['pts3d'][1, 2], (4, 2, 2))
            self.assertFalse(view['valid_mask'][0, 0])
            self.assertTrue(view['sky_mask'][0, 1])
            self.assertTrue(view['is_metric_scale'])
            self.assertEqual(view['pts3d'].dtype, np.float32)

    def test_rejects_resolution_mismatches(self):
        with self.assertRaisesRegex(AssertionError, 'one resolution'):
            SyntheticIGN(ROOT='unused', resolution=[(8, 4), (8, 8)])
        dataset = self.dataset()
        with self.assertRaises(AssertionError):
            dataset[0, 1]
        views = dataset._get_views(0, (8, 4), np.random.default_rng(0))
        views[0]['img'] = Image.new('RGB', (4, 4))
        with patch.object(dataset, '_get_views', return_value=views):
            with self.assertRaisesRegex(AssertionError, 'image size'):
                dataset[0]
        views[0]['img'] = Image.new('RGB', (8, 4))
        views[0]['depthmap'] = np.ones((4, 4), dtype=np.float32)
        with patch.object(dataset, '_get_views', return_value=views):
            with self.assertRaisesRegex(AssertionError, 'Depth resolution'):
                dataset[0]

    def test_matching_contract(self):
        dataset = self.dataset(n_corres=2)
        views = dataset[0]
        self.assertEqual(views[0]['corres'].shape, (2, 2))
        views = dataset._get_views(0, (8, 4), np.random.default_rng(0))
        del views[0]['corres']
        with patch.object(dataset, '_get_views', return_value=views):
            with self.assertRaisesRegex(AssertionError, 'matching targets'):
                dataset[0]

    def test_default_collation_and_dataset_operators(self):
        dataset = self.dataset(n_corres=2)
        loader = get_data_loader(dataset, batch_size=2, num_workers=0, pin_mem=False)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch[0]['img'].shape), (2, 3, 4, 8))
        self.assertEqual(tuple(batch[1]['pts3d'].shape), (2, 4, 8, 3))
        resized = 6 @ dataset
        resized.set_epoch(0)
        self.assertEqual(len(resized), 6)
        self.assertEqual(len(resized[0, 0]), 2)
        combined = dataset + dataset
        self.assertEqual(len(combined), 8)
        self.assertEqual(len(combined[4, 0]), 2)

    def test_storage_hooks_are_explicit(self):
        with self.assertRaisesRegex(NotImplementedError, 'metadata'):
            ign_aerial_ground(ROOT='unused', resolution=(8, 4))
        with self.assertRaisesRegex(NotImplementedError, 'camera loading'):
            ign_aerial_ground._get_views(self.dataset(), 0, (8, 4), np.random.default_rng(0))

    def test_training_namespace_registration(self):
        import dust3r.datasets
        from mast3r.datasets import ign_aerial_ground as exported_dataset

        train_path = Path(__file__).resolve().parents[1] / 'train.py'
        tree = ast.parse(train_path.read_text())
        registration = [statement for statement in tree.body
                if isinstance(statement, ast.Assign)
                and any(ast.unparse(target) == 'dust3r.datasets.ign_aerial_ground'
                    for target in statement.targets)]
        self.assertEqual(len(registration), 1)
        exec(compile(ast.Module(body=registration, type_ignores=[]), str(train_path), 'exec'),
             {'dust3r': __import__('dust3r'), 'ign_aerial_ground': exported_dataset})
        self.assertIs(exported_dataset, ign_aerial_ground)
        self.assertIs(dust3r.datasets.ign_aerial_ground, ign_aerial_ground)
        with patch.object(ign_aerial_ground, '_load_pairs', return_value=[0, 1]), \
                patch.object(ign_aerial_ground, '_get_views', SyntheticIGN._get_views):
            loader = dust3r.datasets.get_data_loader(
                "2 @ ign_aerial_ground(ROOT='unused', resolution=(8,4), seed=0)",
                batch_size=2, num_workers=0, pin_mem=False)
            loader.dataset.set_epoch(0)
            batch = next(iter(loader))
        self.assertEqual(tuple(batch[0]['img'].shape), (2, 3, 4, 8))

    def test_training_entry_point_import(self):
        train_path = Path(__file__).resolve().parents[1] / 'train.py'
        spec = importlib.util.spec_from_file_location('ign_training_entry_point', train_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as error:
            self.skipTest(f'Training dependency unavailable: {error.name}')
        self.assertIs(module.dust3r.datasets.ign_aerial_ground, ign_aerial_ground)


if __name__ == '__main__':
    unittest.main()