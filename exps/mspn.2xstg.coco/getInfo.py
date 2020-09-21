from config import cfg
import random
import cv2
import json
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from google.colab.patches import cv2_imshow
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


if __name__ == "__main__":
    info_file_path = cfg.INFO_PATH
    img_folders_path = cfg.IMG_FOLDER
    res_folders_path = cfg.PRESENT_DIR
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    if os.path.exists(info_file_path):
        os.remove(info_file_path)
    for img_folder in os.listdir(img_folders_path):
        img_folder_abs_path = os.path.join(img_folders_path, img_folder)
        res_folder_abs_path = os.path.join(res_folders_path, img_folder)
        if not os.path.exists(res_folder_abs_path):
            os.makedirs(res_folder_abs_path)
        for img in os.listdir(img_folder_abs_path):
            img_abs_path = os.path.join(img_folder_abs_path, img)
            print("process file:{}".format(img_abs_path))
            im = cv2.imread(img_abs_path)
            outputs = predictor(im)
            person_instances = outputs["instances"][outputs["instances"].pred_classes == 0]
            # 有视频中没有人的情况
            if len(person_instances) == 0:
                no_person = [0, 0, 0, 0]
                with open(info_file_path, 'a') as f:
                    f.write(img_abs_path)
                    f.write(':')
                    f.write(str(no_person))
                    f.write('\n')
                continue
            # 筛选出击球人，目前主要是根据bbox大小来判断
            target_index = 0
            max_area = 0
            for index, box in enumerate(person_instances.pred_boxes):
                width = box[2] - box[0]
                height = box[3] - box[1]
                area = width * height
                target_ndex = index if area > max_area else target_index
            with open(info_file_path, 'a') as f:
                f.write(img_abs_path)
                f.write(':')
                # count = 0
                # for box in person_instances.pred_boxes:
                #     count += 1
                #     box_array = box.cpu().numpy().tolist()
                #     box_adjust = []
                #     width = box_array[2] - box_array[0]
                #     height = box_array[3] - box_array[1]
                #     box_adjust.append(box_array[0])
                #     box_adjust.append(box_array[1])
                #     box_adjust.append(width)
                #     box_adjust.append(height)

                #     f.write(str(box_adjust))
                #     if count == len(person_instances.pred_boxes):
                #         break
                #     f.write('-')
                # f.write('\n')

                for box in person_instances[target_index].pred_boxes:
                    box_array = box.cpu().numpy().tolist()
                    box_adjust = []
                    width = box_array[2] - box_array[0]
                    height = box_array[3] - box_array[1]
                    box_adjust.append(box_array[0])
                    box_adjust.append(box_array[1])
                    box_adjust.append(width)
                    box_adjust.append(height)
                    f.write(str(box_adjust))
                    f.write('\n')
