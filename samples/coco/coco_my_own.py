"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import tensorflow as tf
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
#ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"

############################################################
#  Configurations
############################################################


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.opi
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, return_coco=False,):
        annotation_filepath = os.path.join(dataset_dir, subset, "annotations/annotations.csv")
        dict_annotations = self.read_dict_from_csv(annotation_filepath)
        # 91c9d9c38,"['data/train/imgs/91c9d9c38.jpg', '91c9d9c38.jpg', 1024, 1024, [{'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [124.0, 273.0, 59.0, 73.0], 'category_id': 1, 'id': 0}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [688.0, 939.0, 61.0, 85.0], 'category_id': 1, 'id': 1}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [639.0, 674.0, 83.0, 41.0], 'category_id': 1, 'id': 2}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [562.0, 410.0, 96.0, 83.0], 'category_id': 1, 'id': 3}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [665.0, 92.0, 96.0, 78.0], 'category_id': 1, 'id': 4}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [0.0, 317.0, 103.0, 138.0], 'category_id': 1, 'id': 5}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [681.0, 260.0, 90.0, 57.0], 'category_id': 1, 'id': 6}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [621.0, 430.0, 117.0, 71.0], 'category_id': 1, 'id': 7}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [291.0, 28.0, 75.0, 69.0], 'category_id': 1, 'id': 8}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [687.0, 523.0, 69.0, 100.0], 'category_id': 1, 'id': 9}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [822.0, 532.0, 109.0, 69.0], 'category_id': 1, 'id': 10}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [0.0, 178.0, 93.0, 81.0], 'category_id': 1, 'id': 11}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [41.0, 428.0, 92.0, 112.0], 'category_id': 1, 'id': 12}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [61.0, 443.0, 79.0, 109.0], 'category_id': 1, 'id': 13}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [181.0, 620.0, 85.0, 56.0], 'category_id': 1, 'id': 14}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [0.0, 706.0, 84.0, 103.0], 'category_id': 1, 'id': 15}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [77.0, 717.0, 63.0, 98.0], 'category_id': 1, 'id': 16}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [314.0, 828.0, 49.0, 82.0], 'category_id': 1, 'id': 17}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [154.0, 905.0, 74.0, 85.0], 'category_id': 1, 'id': 18}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [25.0, 943.0, 110.0, 81.0], 'category_id': 1, 'id': 19}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [96.0, 954.0, 90.0, 70.0], 'category_id': 1, 'id': 20}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [686.0, 836.0, 79.0, 58.0], 'category_id': 1, 'id': 21}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [923.0, 378.0, 58.0, 46.0], 'category_id': 1, 'id': 22}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [568.0, 718.0, 106.0, 209.0], 'category_id': 1, 'id': 23}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [50.0, 61.0, 84.0, 61.0], 'category_id': 1, 'id': 24}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [24.0, 122.0, 88.0, 53.0], 'category_id': 1, 'id': 25}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [919.0, 274.0, 53.0, 84.0], 'category_id': 1, 'id': 26}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [130.0, 360.0, 70.0, 58.0], 'category_id': 1, 'id': 27}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [136.0, 553.0, 84.0, 47.0], 'category_id': 1, 'id': 28}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [153.0, 819.0, 87.0, 70.0], 'category_id': 1, 'id': 29}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [283.0, 819.0, 49.0, 35.0], 'category_id': 1, 'id': 30}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [688.0, 784.0, 88.0, 58.0], 'category_id': 1, 'id': 31}, {'area': 1048576, 'iscrowd': 0, 'image_id': '91c9d9c38', 'bbox': [1.0, 0.0, 75.0, 44.0], 'category_id': 1, 'id': 32}]]"

        image_ids = dict_annotations.keys()

        # image_dir:  data/coco//val2017
        # class_ids:  [1, 2, 3, 4, 5, 6, 7, 8, 9,...]

        # image_ids:  [532481, 458755, 245764, 385029, 311303,...]
        # class_ids:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, ...]
        # len(image_ids) =  4952
        # len(class_ids) =  80
        # Add classes
        self.class_info = self.read_label_file(os.path.join(dataset_dir,"label.csv"))

        # Add images
        for id in image_ids:
            self.add_image(
                "coco", image_id=id,
                path= dict_annotations[id][0], # coco.imgs[i]['file_name'] = 000000532481.jpg
                width=dict_annotations[id][2],   # width:   640
                height=dict_annotations[id][3], # height:  427
                annotations=dict_annotations[id][4])
            # annotations:  [{'segmentation': [[253.85, 187.23, ... 254.12, 186.96]], 'area': 2188.086499, 'iscrowd': 0, 'image_id': 532481, 'bbox': [250.82, 168.26, 70.11, 64.88], 'category_id': 1, 'id': 508910}, {'segmentation': [[446.65, 301.37, ..., 448.48, 301.04]], 'area': 82.66089, 'iscrowd': 0, 'image_id': 532481, 'bbox': [435.35, 294.23, 13.46, 7.81], 'category_id': 3, 'id': 1342996}, ...]
        if return_coco:
            return coco
    def read_dict_from_csv(self, csv_path):
        import csv
        import ast
        dict_return = {}
        with open(csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                id = row[0]
                value = ast.literal_eval(row[1])
                dict_return[id] = value
        return dict_return
    def read_label_file(self, label_path):
        import csv
        dict_label = []
        with open(label_path, 'r') as file:
            reader = list(csv.reader(file))
            for row in reader[1:]:
                id, source, name = row
                dict_label.append({'source': source, 'id': id, 'name': name})
        return dict_label

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)
        # image_info:  {'id': 373705, 'source': 'coco', 'path': 'data/coco//val2017/000000373705.jpg', 'width': 640, 'height': 427, 'annotations': [{'segmentation': [[492.13, 186.88, 494.67, 183.72, 501.0, 182.45, 507.96, 179.28, 512.39, 174.85, 517.46, 167.26, 522.52, 157.13, 524.42, 148.9, 525.69, 136.24, 525.69, 126.11, 528.85, 113.45, 530.75, 110.28, 530.75, 107.75, 527.59, 102.68, 523.79, 96.35, 523.79, 93.19, 523.79, 90.66, 521.25, 80.53, 521.89, 76.73, 527.59, 69.13, 532.02, 67.86, 537.71, 65.33, 540.88, 65.33, 552.91, 70.4, 556.71, 77.99, 556.71, 79.89, 556.07, 91.29, 556.07, 95.09, 556.07, 96.99, 559.24, 101.42, 565.57, 114.08, 568.1, 118.51, 571.27, 124.84, 575.7, 134.34, 576.33, 138.14, 578.23, 147.63, 578.86, 164.72, 579.5, 171.05, 579.5, 179.92, 579.5, 183.72, 579.5, 190.68, 579.5, 195.11, 574.43, 205.87, 571.9, 210.31, 567.47, 210.94, 561.77, 210.94, 556.71, 210.94, 553.54, 210.31, 549.74, 205.87, 546.58, 199.54, 541.51, 196.38, 541.51, 195.11, 541.51, 192.58, 541.51, 189.41, 541.51, 188.15, 533.28, 183.72, 528.85, 188.78, 528.85, 190.68, 519.99, 195.74, 518.09, 195.74, 515.56, 195.74, 513.66, 195.74, 502.9, 189.41, 497.83, 188.15]], 'area': 6544.306350000003, 'iscrowd': 0, 'image_id': 373705, 'bbox': [492.13, 65.33, 87.37, 145.61], 'category_id': 1, 'id': 460031}, {'segmentation': [[230.29, 146.81, 237.51, 148.13, 245.87, 156.91, 245.45, 166.93, 249.63, 181.97, 248.37, 191.99, 250.04, 202.43, 233.76, 201.18, 232.92, 190.74, 225.82, 190.32, 222.48, 193.66, 219.14, 188.65, 219.14, 172.78, 214.96, 180.3, 210.78, 178.62, 211.2, 186.14, 207.86, 192.83, 206.61, 184.05, 207.86, 153.15, 213.29, 148.97, 209.95, 139.78, 211.62, 133.93, 220.39, 129.76, 226.66, 132.26, 228.74, 138.11, 228.74, 142.71, 230.41, 148.13]], 'area': 2006.234, 'iscrowd': 0, 'image_id': 373705, 'bbox': [206.61, 129.76, 43.43, 72.67], 'category_id': 1, 'id': 465356}, {'segmentation': [[106.51, 108.27, 103.63, 99.63, 103.63, 91.0, 112.27, 86.2, 122.82, 85.24, 127.62, 88.12, 125.7, 98.67, 118.98, 109.23, 123.78, 117.86, 126.66, 129.38, 134.34, 153.37, 136.26, 170.64, 136.26, 178.32, 136.26, 179.28, 130.5, 176.4, 122.82, 143.77, 120.9, 162.0, 124.74, 194.63, 116.11, 200.39, 105.55, 201.35, 95.96, 194.63, 94.04, 140.89, 95.0, 121.7, 102.67, 114.99]], 'area': 3240.2072500000004, 'iscrowd': 0, 'image_id': 373705, 'bbox': [94.04, 85.24, 42.22, 116.11], 'category_id': 1, 'id': 468751}, {'segmentation': [[133.28, 203.84, 130.95, 185.79, 142.02, 173.55, 143.77, 171.22, 139.69, 163.65, 141.44, 156.66, 149.59, 154.33, 156.0, 157.24, 157.75, 165.98, 157.16, 171.22, 155.75, 174.66, 164.59, 183.5, 164.95, 193.75, 166.01, 204.36]], 'area': 1179.02015, 'iscrowd': 0, 'image_id': 373705, 'bbox': [130.95, 154.33, 35.06, 50.03], 'category_id': 1, 'id': 542379}, {'segmentation': [[0.78, 186.44, 2.46, 173.49, 4.38, 173.25, 7.5, 170.97, 9.42, 166.89, 10.14, 164.49, 6.78, 160.78, 2.82, 159.46, 0.06, 162.7]], 'area': 116.41799999999998, 'iscrowd': 0, 'image_id': 373705, 'bbox': [0.06, 159.46, 10.08, 26.98], 'category_id': 1, 'id': 1259641}, {'segmentation': [[638.29, 208.62, 623.96, 207.79, 620.1, 186.85, 623.41, 171.42, 626.17, 163.42, 622.31, 152.4, 623.69, 148.54, 623.96, 144.96, 632.23, 138.35, 634.71, 135.59, 639.95, 135.59, 639.95, 202.55]], 'area': 1172.0853500000032, 'iscrowd': 0, 'image_id': 373705, 'bbox': [620.1, 135.59, 19.85, 73.03], 'category_id': 1, 'id': 1709692}, {'segmentation': [[497.64, 424.89, 494.48, 251.98, 500.8, 292.05, 509.24, 313.13, 521.89, 318.4, 524.0, 306.81, 534.54, 304.7, 535.6, 293.1, 538.76, 282.56, 545.08, 274.12, 541.92, 261.47, 541.92, 253.04, 549.3, 240.39, 551.41, 236.17, 555.63, 221.41, 547.19, 204.54, 538.76, 195.05, 539.81, 185.56, 528.21, 185.56, 526.11, 198.21, 517.67, 198.21, 512.4, 201.38, 502.91, 195.05, 498.69, 191.89, 489.2, 194.0, 481.82, 180.29, 463.9, 168.69, 458.63, 165.53, 451.25, 166.58, 447.03, 171.85, 424.89, 185.56, 418.57, 197.16, 403.8, 196.1, 395.37, 197.16, 390.1, 196.1, 382.72, 199.27, 373.23, 201.38, 366.9, 221.41, 365.85, 231.95, 376.39, 242.49, 393.26, 276.23, 400.64, 285.72, 406.97, 299.43, 419.62, 299.43, 422.78, 300.48, 427.0, 303.64, 427.0, 423.84]], 'area': 27177.1407, 'iscrowd': 0, 'image_id': 373705, 'bbox': [365.85, 165.53, 189.78, 259.36], 'category_id': 11, 'id': 1808923}]}

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # annotations:  [{'segmentation': [[214.27, 79.01,... 212.6, 78.59]], 'area': 970.04309, 'iscrowd': 0, 'image_id': 262145, 'bbox': [212.6, 55.62, 53.04, 53.46], 'category_id': 28, 'id': 284647}, {'segmentation': [[35.5, 1.76, ... 202.47, 3.68]], 'area': 6085.1964499999995, 'iscrowd': 0, 'image_id': 262145, 'bbox': [13.43, 1.76, 226.46, 55.65], 'category_id': 28, 'id': 285569}, ...]

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            # class_id:  1
            if class_id:
                width = image_info['width']
                height = image_info['height']
                binary_mask = np.zeros(shape = (height, width))

                x1,y1,w,h = annotation['bbox']
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x1+w)
                y2 = int(y1+h)
                for i in range(y1,y2):
                    for j in range(x1,x2):
                        binary_mask[i][j] = 1

                """
                binary_mask:  [[0. 0. 0. ... 0. 0. 0.]
                               [0. 0. 0. ... 0. 0. 0.]
                               ...
                               [0. 0. 0. ... 0. 0. 0.]]
                binary_mask.shape =  (1024, 1024)
                distinct elelment of binary_mask:  {0.0, 1.0}
                """
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if binary_mask.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if binary_mask.shape[0] != image_info["height"] or binary_mask.shape[1] != image_info["width"]:
                        binary_mask = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(binary_mask)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",  metavar="<command>", help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True, metavar="data/coco", help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False, default=DEFAULT_DATASET_YEAR, metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=True, metavar="/path/to/weights.h5", help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--loadmodel', dest='bool_loadmodel', action='store_true')
    parser.add_argument('--no_loadmodel', dest='bool_loadmodel', action='store_false')
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR,  metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False, default=500, metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False, default=False, metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)', type=bool)
    args = parser.parse_args()

    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model
    if args.bool_loadmodel == True:
        # Load weights
        print("load weights")
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        #dataset_train.load_coco(args.dataset, "train", year=args.year)
        dataset_train.load_coco(args.dataset, subset= "train")
        dataset_train.prepare()
        #print("image_info: ", dataset_train.image_info)
        dataset_train.load_mask(0)

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_coco(args.dataset, subset= "val" )
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=90,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=120,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val"
        coco = dataset_val.load_coco(args.dataset, subset=val_type)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))

# python samples/coco/coco_my_own.py train --dataset=data --model=coco --no_loadmodel

# python samples/coco/coco_my_own.py train --dataset=data --model=logs/coco20210113T1106/mask_rcnn_coco_0210.h5 --loadmodel

# python samples/coco/coco_my_own.py evaluate --dataset=data --model=logs/coco20210113T1106/mask_rcnn_coco_0210.h5 --loadmodel
