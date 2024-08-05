import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import torchvision.transforms as transforms
from mmseg.apis import init_model, inference_model
import mmcv

class DLSAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self):
        # self.deeplab = init_model(self.cfg.deeplab.config_path, self.cfg.deeplab.checkpoint)
        # self.deeplab.train()
        # if self.cfg.deeplab.freeze.backbone:
        #     for param in self.deeplab.backbone.parameters():
        #         param.requires_grad = False
        #     for param in self.deeplab.decode_head.parameters():
        #         param.requires_grad = False
        #     for param in self.deeplab.auxiliary_head.parameters():
        #         param.requires_grad = False
        
        self.sam = sam_model_registry[self.cfg.sam.type](checkpoint=self.cfg.sam.checkpoint)
        #self.sam.image_encoder.img_size = 256
        self.sam.train()
        if self.cfg.sam.freeze.image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.sam.freeze.prompt_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.sam.freeze.mask_decoder:
            for param in self.sam.freeze.mask_decoder:
                param.requires_grad = False

    def forward(self, images, pre_masks):
        _, _, H, W = images.shape
        #preprocess
        target_size = self.sam.image_encoder.img_size
        
        transform = ResizeLongestSide(target_size)
        images = transform.apply_image_torch(images)
        #images = torch.stack([self.preprocess(x) for x in torch.unbind(images, dim=0)], dim=0)
        
        #image_encoder
        image_embeddings = self.sam.image_encoder(images)
        pred_masks = []
        ious = []
        for embedding, pre_mask in zip(image_embeddings, pre_masks):
            #prompt_encoder with pre_masks
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=pre_mask,
            )

            #mask_decoder
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.sam)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """    
        To transform the input image, label and low resolution mask
        into the input shape as image_encoder and prompt_encoder required
        including resize and pad 
        """

        #pad image and mask to a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)

        return image
       
        



