import os
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

register_coco_instances(
    "marine_train",
    {},
    "/workspace/dataset/train/_annotations.coco.json",
    "/workspace/dataset/train"
)

register_coco_instances(
    "marine_val",
    {},
    "/workspace/dataset/valid/_annotations.coco.json",
    "/workspace/dataset/valid"
)

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)

cfg.DATASETS.TRAIN = ("marine_train",)
cfg.DATASETS.TEST = ("marine_val",)

cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 40000

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

cfg.OUTPUT_DIR = "/workspace/output"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
