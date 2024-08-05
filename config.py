from box import Box

config = {
    "num_devices": 1,
    'batch_size': 2,
    "num_workers": 4,
    "num_epochs": 20,
    "eval_interval": 10,
    "out_dir": "/root/xyf/semantic segmentation MMseg/lightning/out/training",
    "opt":{
        "learning_rate": 8e-4,  #8e-4
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps":250,
    },
    "sam":{
        "type": 'vit_h',
        "checkpoint": "/root/xyf/sam_vit_h_4b8939.pth",
        "freeze":{
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False
        },
    },
    "deeplab":{
        "type": "Deeplabv3",
        "checkpoint": '/root/xyf/semantic segmentation MMseg/mmsegmentation/work_dirs/deeplabv3_r50-d8_4xb4-80k_AI4b-256x256/iter_80000.pth',
        "config_path": '/root/xyf/semantic segmentation MMseg/mmsegmentation/work_dirs/deeplabv3_r50-d8_4xb4-80k_AI4b-256x256/deeplabv3_r50-d8_4xb4-80k_AI4b-256x256.py',
        "freeze":{
            "backbone": True,
            "decode_head": True,
            "auxiliary_head": True
        }
    },
    "dataset":{
        "images": {
            "training": "/root/xyf/semantic segmentation MMseg/mmsegmentation/AI4B_data/AI4boundaries_dataset/AI4B_preprocessed/images/training",
            "validation": "/root/xyf/semantic segmentation MMseg/mmsegmentation/AI4B_data/AI4boundaries_dataset/AI4B_preprocessed/images/validation"
        },
        "annotations":{
            "training":"/root/xyf/semantic segmentation MMseg/mmsegmentation/AI4B_data/AI4boundaries_dataset/AI4B_preprocessed/annotations/training",
            "validation": "/root/xyf/semantic segmentation MMseg/mmsegmentation/AI4B_data/AI4boundaries_dataset/AI4B_preprocessed/annotations/validation"
        },
        "premasks":{
            "training": "/root/xyf/semantic segmentation MMseg/mmsegmentation/AI4B_data/AI4boundaries_dataset/AI4B_preprocessed/premasks/training",
            "validation": "/root/xyf/semantic segmentation MMseg/mmsegmentation/AI4B_data/AI4boundaries_dataset/AI4B_preprocessed/premasks/validation"
        }
    }
}

cfg = Box(config)