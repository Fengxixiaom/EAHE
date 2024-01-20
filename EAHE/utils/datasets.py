import os
import cv2
import numpy as np
import random
import math
import torch

from torch.utils.data import Dataset
from utils.test_generate import generate_dataset
from utils.extract_edge import get_edge_map
from utils.general import make_divisible
from utils.torch_utils import torch_distributed_zero_first


def cross(a, b, c):
    ans = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    return ans
def checkShape(a, b, c, d):
    x1 = cross(a, b, c)
    x2 = cross(b, c, d)
    x3 = cross(c, d, a)
    x4 = cross(d, a, b)

    if (x1 < 0 and x2 < 0 and x3 < 0 and x4 < 0) or (x1 > 0 and x2 > 0 and x3 > 0 and x4 > 0):
        return 1
    else:
        print('not convex')
        return 0

def create_dataloader(path, imgsz, batch_size, mode='align', reg_mode='resize',
                      hyp=None, augment=False, rank=-1, world_size=1, workers=8):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagePairs(path, imgsz, augment=augment, hyp=hyp, mode=mode, reg_mode=reg_mode)
    # print("dataset",type(dataset))
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    # loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    loader = InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset, batch_size=batch_size, shuffle=augment,
                        num_workers=nw, sampler=sampler, pin_memory=True,
                        collate_fn=CollateFn(mode_align=mode == 'align')
                        )
    
    return dataloader, dataset

class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def BGR_to_RGB(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg
class CollateFn(object):
    def __init__(self, mode_align=True):
        self.mode_align = mode_align
    
    def __call__(self, batch):
        shapes = np.array([img.shape[:2] for img, img1_warped_mask, img1_warped, offset in batch])
        hm, wm = [make_divisible(int(shapes[:, i].max()), 32) for i in range(2)]
        if self.mode_align:
            hm, wm = max(hm, wm), max(hm, wm)  # square

        outs = []
        offset_list = []
        img1_warped_list = []
        img1_warped_mask_list = []
        for i, img_ in enumerate(batch):
            h, w = shapes[i]
            dx, dy = (wm - w) // 2, (hm - h) // 2
            img1_warped_mask, img1_warped, offset = img_[1], img_[2], img_[3]
            img = img_[0]
            # print("img1_warped_mask",img1_warped_mask.shape)
            # img1_warped_mask = img1_warped_mask[...,np.newaxis]
            offset = offset[...,np.newaxis]

            img1_warped_list.append(np.pad(img1_warped, ((dy, hm - h - dy), (dx, wm - w - dx), (0, 0))))
            img1_warped_mask_list.append(np.pad(img1_warped_mask.squeeze(), ((dy, hm - h - dy), (dx, wm - w - dx), (0, 0))))
            offset_list.append(offset)
            outs.append(np.pad(img, ((dy, hm - h - dy), (dx, wm - w - dx), (0, 0))))  # h,w,c
            
        imgs = np.stack(outs)
        img1_warped_mask_list = np.stack(img1_warped_mask_list)
        img1_warped_list = np.stack(img1_warped_list)
        offset = np.stack(offset_list)
        imgs = imgs[..., ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, bhwc to bchw
        img1_warped_list = img1_warped_list[..., ::-1].transpose(0, 3, 1, 2)
        img1_warped_list = np.ascontiguousarray(img1_warped_list)
        imgs = np.ascontiguousarray(imgs)
        return torch.from_numpy(imgs), torch.from_numpy(img1_warped_mask_list).permute(0, 3, 1, 2), torch.from_numpy(img1_warped_list), torch.tensor(offset)

class LoadImagePairs(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, augment=False, hyp=None, mode='align', reg_mode='resize'):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.mode = mode
        self.path = path
        self.reg_mode = reg_mode  # `resize` or `crop`. Regularize the input images.
        
        with open(path, 'r') as f:
            self.img_files = f.read().splitlines()
        self.indices = range(len(self.img_files))  # number of images
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        # Load image
        img1, img2, msk1, msk2, img1_warped, img1_warped_mask, offset = self.load_data_(self.img_files[index])
        hyp = self.hyp
        img1_warped_mask = img1_warped_mask[..., np.newaxis]
        image = np.concatenate((msk1, img1, msk2, img2), axis=-1)
        return image, img1_warped_mask, img1_warped, offset  # ch=1+3+1+3=8, hwc

    def load_data_(self, path):
        img1 = cv2.imread(path)
        img1_ = img1
        offset, h = generate_dataset(img1_)
        img2 = cv2.imread(path.replace('input1', 'input2'))
        new_size = (self.img_size, self.img_size)
        if tuple(img1.shape[:2]) != new_size or tuple(img2.shape[:2]) != new_size:
            interpolation = cv2.INTER_AREA  # cv2.INTER_LINEAR cv2.INTER_AREA
            img1 = cv2.resize(img1, new_size, interpolation=interpolation)
            img2 = cv2.resize(img2, new_size, interpolation=interpolation)

        msk1 = np.zeros((*new_size, 1), dtype=np.uint8) + 255
        msk2 = np.zeros((*new_size, 1), dtype=np.uint8) + 255
        img1_ = cv2.GaussianBlur(img1, ksize=(5, 5), sigmaX=0, sigmaY=0)
        img1_warped = get_edge_map(img1_)

        img1_warped = np.stack((img1_warped,) * 3, axis=-1)

        img2_ = cv2.GaussianBlur(img2, ksize=(5, 5), sigmaX=0, sigmaY=0)
        img1_warped_mask = get_edge_map(img2_)
        img1_warped_mask = np.stack((img1_warped_mask,) * 3, axis=-1)

        return img1, img2, msk1, msk2, img1_warped, img1_warped_mask, offset



def random_perspective(image, mask=None, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):

    height = image.shape[0] + border[0] * 2  # shape(h,w,c)
    width = image.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(0, 0, 0))
            mask = cv2.warpPerspective(mask, M, dsize=(width, height), borderValue=(0, 0, 0)) \
                if mask is not None else None
        else:  # affine
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
            mask = cv2.warpAffine(mask, M[:2], dsize=(width, height), borderValue=(0, 0, 0)) \
                if mask is not None else None

    return (image, mask) if mask is not None else image


def augment_hsv(img, img2=None, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
    
    if img2 is not None:
        hue, sat, val = cv2.split(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV))
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img2)  # no return needed


def color_balance(img1, img2, msk1=None, msk2=None, normalized=False):
    if msk1 is None or msk2 is None:
        msk1 = np.ones_like(img1[..., :1])
        msk2 = np.ones_like(img2[..., :1])
    if normalized:
        img1, img2 = img1 * 255., img2 * 255.

    idx1, idx2 = msk1[..., 0] > 0, msk2[..., 0] > 0
    r = ((img2[idx2] / 255.).mean(axis=0) / (img1[idx1] / 255.).mean(axis=0)).reshape(1, 1, 3)
    img1 = (((img1 / 255.) * r).clip(0, 1) * 255.).astype(np.uint8)
    return img1
