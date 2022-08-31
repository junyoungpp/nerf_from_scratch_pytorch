import math
import torch
import unittest
import nerf_dataloader
import nerf_mlp

import pdb

class TestDataLoader(unittest.TestCase):
    def test_load_transforms(self):
        filepath = './data/fox/transforms.json'
        dataset = nerf_dataloader.NeRFDataset(filepath, load_images=False)
        self.assertTrue(len(dataset) > 0)

    def test_get_rays(self):
        filepath = './data/fox/transforms.json'
        dataset = nerf_dataloader.NeRFDataset(filepath, load_images=False)
        uv, ray_origin, ray_dir, rgb = nerf_dataloader.NeRFDataset.get_rays(dataset.transforms['camera'], dataset[0], 2048)
        self.assertEqual(uv.shape[0], 2048)
        self.assertEqual(uv.shape[1], 2)
        self.assertEqual(ray_origin.shape[0], 2048)
        self.assertEqual(ray_origin.shape[1], 3)
        self.assertEqual(ray_dir.shape[0], 2048)
        self.assertEqual(ray_dir.shape[1], 3)
        self.assertEqual(rgb.shape[0], 2048)
        self.assertEqual(rgb.shape[1], 3)


class TestNeRFMLP(unittest.TestCase):
    def test_frequency_encoding(self):
        xyz = torch.rand(1, 3)
        ret = nerf_mlp.NeRFMLP.frequency_encoding(xyz, 10)
        self.assertEqual(ret.shape[1], 60)

        x = xyz[0][0]
        y = xyz[0][1]
        z = xyz[0][2]
        EPS = 1e-12
        for i in range(10):
            weight = pow(2.0, i)
            self.assertTrue(math.fabs(ret[0, 6 * i + 0] - torch.sin(torch.Tensor([weight * x]))) < EPS)
            self.assertTrue(math.fabs(ret[0, 6 * i + 1] - torch.sin(torch.Tensor([weight * y]))) < EPS)
            self.assertTrue(math.fabs(ret[0, 6 * i + 2] - torch.sin(torch.Tensor([weight * z]))) < EPS)

            self.assertTrue(math.fabs(ret[0, 6 * i + 3] - torch.cos(torch.Tensor([weight * x]))) < EPS)
            self.assertTrue(math.fabs(ret[0, 6 * i + 4] - torch.cos(torch.Tensor([weight * y]))) < EPS)
            self.assertTrue(math.fabs(ret[0, 6 * i + 5] - torch.cos(torch.Tensor([weight * z]))) < EPS)

    def test_nerf_mlp(self):
        xyz = torch.rand(10, 3)
        dir = torch.rand(10, 3)
        mlp = nerf_mlp.NeRFMLP()
        density, color = mlp(xyz, dir)
        self.assertEqual(density.shape[0], 10)
        self.assertEqual(color.shape[0], 10)
        self.assertEqual(color.shape[1], 3)

if __name__ == '__main__':
    unittest.main()
