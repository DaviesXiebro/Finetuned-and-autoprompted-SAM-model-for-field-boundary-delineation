import os

import cv2
import torch
from box import Box
from model import DLSAM
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm

from mmseg.apis import init_model, inference_model
import mmcv


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if boxes is not None:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if masks is not None:
        image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)


def visualize(cfg: Box):
    model = Model(cfg)
    model.setup()
    model.eval()
    model.cuda()
    dataset = AI4bDataset(cfg=cfg,
                      img_dir=cfg.dataset.images.validation,
                      gt_dir=cfg.dataset.annotations.validation,
                      transform=transform)
    predictor = model.get_predictor()
    os.makedirs(cfg.out_dir, exist_ok=True)
    deeplab = init_model(cfg.deeplab.config_path, cfg.deeplab.checkpoint, 'cpu')
    
    for image_name in dataset.img_file_list:
        output_name = os.path.basename(image_name)
        image_path = image_name
        image_output_path = os.path.join(cfg.out_dir, output_name)
        img_gadl = gdal.Open(image_path)
        img_gadl = img_gadl.ReadAsArray(band_list=band_list)
        img_gadl = img_gadl.transpose(1,2,0)
        img_gadl = img_gadl.astype(np.uint8)

        result = inference_model(deeplab, img_name)
        logits = result.seg_logits.data[1,:,:]
        mask = logits[None, :, :]

        predictor.set_image(img_gadl)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            boxes=None,
            mask_input=mask,
            multimask_output=False,
        )
        image_output = draw_image(img_gadl, masks.squeeze(1), boxes=None, labels=None)
        cv2.imwrite(image_output_path, image_output)


if __name__ == "__main__":
    from config import cfg
    visualize(cfg)