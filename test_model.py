import cv2
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
cfg.MODEL.WEIGHTS = "/workspace/output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda"

predictor = DefaultPredictor(cfg)

image_path = "/workspace/dataset/test"

for img_name in os.listdir(image_path):

    if not img_name.lower().endswith((".jpg",".jpeg",".png")):
        continue

    img = cv2.imread(os.path.join(image_path, img_name))

    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1], scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imwrite(f"/workspace/output/pred_{img_name}", out.get_image()[:, :, ::-1])
