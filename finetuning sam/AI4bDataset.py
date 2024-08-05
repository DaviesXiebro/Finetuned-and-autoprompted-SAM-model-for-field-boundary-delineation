import os
from osgeo import gdal
import numpy as np
import torch
import torchvision.transforms as transforms
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from mmseg.apis import init_model, inference_model
import mmcv

class AI4bDataset(Dataset):
    def __init__(self, cfg, img_dir, gt_dir, pm_dir, transform=None):
        super().__init__()
        self.transform = transform
        self.gt_path = gt_dir
        self.img_path = img_dir
        self.pm_path = pm_dir
        self.img_file_list = sorted(os.listdir(self.img_path))
        self.gt_file_list = sorted(os.listdir(self.gt_path))
        self.pm_file_list = sorted(os.listdir(self.pm_path))
 
    def __len__(self):
        return len(self.img_file_list)


    
    def __getitem__(self, idx):
        img_file_path = os.path.join(self.img_path, self.img_file_list[idx])
        gt_file_path = os.path.join(self.gt_path, self.gt_file_list[idx])
        pm_file_path = os.path.join(self.pm_path, self.pm_file_list[idx])
        band_list = [1,2,3]
        band_name = [2] #boundary
        logit_list = [2]
        img_data = read_tiff_image(img_file_path, band_list)
        gt_data = read_tiff_annotation(gt_file_path, band_name)
        mask_data = read_input_mask(pm_file_path, logit_list)

        if self.transform:
            img_data, gt_data = self.transform(img_data, gt_data) 

        return {
            "image": img_data,
            "label": torch.as_tensor(gt_data)[None, :, :],
            "mask": torch.as_tensor(mask_data)[None, :, :]
        }

def read_tiff_image(img_name, band_list):
    img_gadl = gdal.Open(img_name)
    img_gadl = img_gadl.ReadAsArray(band_list=band_list)
    #img_gadl = img_gadl.transpose(1,2,0)
    img_gadl = img_gadl.astype(np.float32)
    return img_gadl

def read_tiff_annotation(ann_name, band_name):
    gt_gadl = gdal.Open(ann_name)
    gt_gadl = gt_gadl.ReadAsArray(band_list=band_name)
    gt_gadl = gt_gadl.astype(np.float32)
    gt_gadl = np.where(gt_gadl==0, gt_gadl, -1)
    return gt_gadl

def read_input_mask(mask_name, logit_list):
    mask_gadl = gdal.Open(mask_name)
    mask_gadl = mask_gadl.ReadAsArray(band_list=logit_list)
    mask_gadl = mask_gadl.astype(np.float32)
    return mask_gadl


def load_datasets(cfg, img_size):
    #transform = Preprocessor(img_size)
    train = AI4bDataset(cfg=cfg,
                        img_dir=cfg.dataset.images.training,
                        gt_dir=cfg.dataset.annotations.training,
                        pm_dir=cfg.dataset.premasks.training)
    val = AI4bDataset(cfg=cfg,
                      img_dir=cfg.dataset.images.validation,
                      gt_dir=cfg.dataset.annotations.validation,
                      pm_dir=cfg.dataset.premasks.validation)
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=cfg.num_workers)
    return train_dataloader, val_dataloader
    

class Preprocessor:
    """    
    To transform the input image, label and low resolution mask
    into the input shape as image_encoder and prompt_encoder required
    including resize and pad 
    """
    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, label):
        #resize the image and mask
        it_h, it_w, _ = image.shape
        image = self.transform.apply_image(image)
        label = torch.tensor(self.transform.apply_image(label))
        image = self.to_tensor(image)

        #pad image and mask to a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        label = transforms.Pad(padding)(label)

        return image, label


