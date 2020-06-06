from data import datasets
import os
import numpy as np
import json
import random
import cv2
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

dir = 'dataset/composite/'
classes = json.load(open(os.path.join(dir, 'train/labels.json')))

#for d in ["train", "val", "test"]:
#    DatasetCatalog.register("composite_" + d, lambda d=d: get_dataset(d))
#    MetadataCatalog.get("composite_" + d).set(thing_classes=classes)

composite_metadata = MetadataCatalog.get("composite_train")
                                                             
def get_dataset(mode):
    filename = os.path.join(dir, mode + '/data.json')
    with open(filename) as f:
        return json.load(f)

cfg = get_cfg()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("composite_val", )
predictor = DefaultPredictor(cfg)

def find_nearest_neighbor(model=predictor):
    dataset_dicts = get_dataset("val")
    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        outputs = model(im)
        print("Starting to print outputs of prediction")
        print(outputs)
        print(outputs["instances"]._fields["pred_boxes"].tensor)
        v = Visualizer(im[:, :, ::-1], metadata=composite_metadata, scale=0.8)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(v.get_image()[:, :, ::-1])

def main():
    find_nearest_neighbor()

if __name__ == "__main__": 
    main()


