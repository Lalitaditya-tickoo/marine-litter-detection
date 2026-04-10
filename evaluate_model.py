from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

register_coco_instances(
    "marine_val",
    {},
    "/workspace/dataset/valid/_annotations.coco.json",
    "/workspace/dataset/valid"
)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
cfg.MODEL.WEIGHTS = "/workspace/output/model_final.pth"
cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("marine_val", cfg, False, output_dir="./output")
val_loader = build_detection_test_loader(cfg, "marine_val")

results = inference_on_dataset(predictor.model, val_loader, evaluator)
print(results)
