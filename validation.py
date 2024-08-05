import os
import time

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from AI4bDataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from loss_functions import DiceLoss
from loss_functions import FocalLoss
from model import DLSAM
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou

torch.set_float32_matmul_precision('high')


def validate(fabric: L.Fabric, model: DLSAM, val_dataloader: DataLoader, epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    accuracy = AverageMeter()
    precision = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images = data["image"]
            labels = data["label"]
            masks = data["mask"]
            num_images = images.size(0)
            pred_masks, _ = model(images, masks)
            for pred_mask, gt_mask in zip(pred_masks, labels):
                batch_stats = smp.metrics.get_stats(
                    pred_mask.int(),
                    gt_mask.int(),
                    mode='binary',
                    threshold=0,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                batch_accuracy = smp.metrics.accuracy(*batch_stats, reduction="micro-imagewise")
                batch_precision = smp.metrics.positive_predictive_value(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
                accuracy.update(batch_accuracy, num_images)
                precision.update(batch_precision, num_images)
            fabric.print(
                f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- Mean accuracy[{accuracy.avg:.4f}] -- Mean precision: [{precision.avg:.4f}]'
            )

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- Mean accuracy[{accuracy.avg:.4f}] -- Mean precision: [{precision.avg:.4f}]')

    # fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    # state_dict = model.sam.state_dict()
    # if fabric.global_rank == 0:
    #     torch.save(state_dict, os.path.join(cfg.out_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    # model.train()

def configure_opt(cfg: Box, model: DLSAM):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.sam.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

def main(cfg: Box) -> None:
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = DLSAM(cfg)
        model.setup()

    train_data, val_data = load_datasets(cfg, model.sam.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    validate(fabric, model, val_data, epoch=0)


if __name__ == "__main__":
    cfg.sam.checkpoint='/root/xyf/semantic segmentation MMseg/lightning/out/training/ftSAM_epoch20_boundary.pth'
    main(cfg)