import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as tf
import numpy as np

import pdb

class NeRFDataset(Dataset):
    def __init__(self, filepath, load_images=False):
        self.filepath = filepath
        self.load_images = load_images
        self.transforms = self.load_transforms(filepath, load_images=load_images)
    
    @staticmethod
    def load_transforms(transforms_filepath, load_images=True):
        transforms_dirpath = os.path.dirname(transforms_filepath)
        with open(transforms_filepath) as f:
            transforms = json.load(f)
            assert transforms['type'] == 'nerf_dataset'
            images = transforms['images']
            for image in images:
                image_filepath = os.path.join(transforms_dirpath, image['file'])
                image['filepath'] = image_filepath
                data = None
                if load_images == True:
                    data = tf.ToTensor()(Image.open(image_filepath))
                image['data'] = data
        return transforms

    @staticmethod
    def distort(uv_norm, camera):
        dists = camera['distortions']
        k1 = dists[0]
        k2 = dists[1]
        k3 = dists[2]
        r2 = uv_norm[:, 0] * uv_norm[:, 0] + uv_norm[:, 1] * uv_norm[:, 1]
        distortion = 1.0 + r2 * (k1 + r2 * k2 + r2 * r2 * k3)
        uv_norm_distorted = uv_norm * distortion.unsqueeze(1)
        #// tangential distorsion
        #xd = xd + (2. * p1 * x * y + p2 * (r2 + 2. * x * x));
        #yd = yd + (p1 * (r2 + 2. * y * y) + 2. * p2 * x * y);

        return uv_norm_distorted

    @staticmethod
    def undistort(uv, camera):
        center_of_pixel = camera['center_of_pixel']
        focal_length = camera['focal_length']
        uv_norm = (uv.to(float) - torch.tensor(center_of_pixel)) / focal_length

        d = NeRFDataset.distort(uv_norm, camera) - uv_norm
        iter_count = 100
        for i in range(iter_count):
            uv_norm_undistorted = uv_norm - d
            d = NeRFDataset.distort(uv_norm, camera) - uv_norm_undistorted

        uv_undistorted = (uv_norm_undistorted * focal_length) + torch.tensor(center_of_pixel)
        return uv_undistorted

    @staticmethod
    def get_rays(camera, image, sample_count):
        image_size = camera['image_size']
        width = image_size[0]
        height = image_size[1]
        u = torch.randint(0, width, (sample_count, 1))
        v = torch.randint(0, height, (sample_count, 1))
        uv = torch.hstack((u, v))
        rgb = None
        if 'data' in image:
            data = image['data']
            rgb = data[:, uv[:,1], uv[:,0]]
            rgb = torch.transpose(rgb, 0, 1)
        center_of_pixel = camera['center_of_pixel']
        focal_length = camera['focal_length']
        ray_dir_cam = (NeRFDataset.undistort(uv.to(float), camera) - torch.tensor(center_of_pixel)) / focal_length
        ray_dir_cam = torch.hstack((ray_dir_cam, torch.ones((2048, 1))))
        ray_dir_cam = torch.nn.functional.normalize(ray_dir_cam)
        tm = image['tm']
        ray_x = tm[0][0] * ray_dir_cam[:,0] + tm[0][1] * ray_dir_cam[:,1] + tm[0][2] * ray_dir_cam[:,2]
        ray_y = tm[1][0] * ray_dir_cam[:,0] + tm[1][1] * ray_dir_cam[:,1] + tm[1][2] * ray_dir_cam[:,2]
        ray_z = tm[2][0] * ray_dir_cam[:,0] + tm[2][1] * ray_dir_cam[:,1] + tm[2][2] * ray_dir_cam[:,2]
        ray_dir_world = torch.hstack((ray_x.unsqueeze(1), ray_y.unsqueeze(1), ray_z.unsqueeze(1)))
        ray_origin_world = torch.tensor([tm[0][3], tm[1][3], tm[2][3]]).expand(ray_dir_world.shape[0], 3)

        return uv, ray_origin_world, ray_dir_world, rgb

    def __len__(self):
        if self.transforms is None:
            return 0
        if not 'images' in self.transforms:
            return 0
        return len(self.transforms['images'])

    def __getitem__(self, idx):
        image = self.transforms['images'][idx]
        if image['data'] is None:
            image_filepath = image['filepath']
            data = tf.ToTensor()(Image.open(image_filepath))
            image['data'] = data
        return image


if __name__ == '__main__':
    filepath = './data/fox/transforms.json'
    dataset = NeRFDataset(filepath, load_images=True)
    print(dataset[0])
    print(len(dataset))
    print(dataset.transforms['camera'])
    print(NeRFDataset.get_rays(dataset.transforms['camera'], dataset[0], 2048))
    
            