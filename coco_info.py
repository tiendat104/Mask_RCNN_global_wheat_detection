
import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

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
ROOT_DIR = os.path.abspath("../../")

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
dataset_dir = "data/coco"

coco = COCO("data/coco/annotations/instances_val2017.json")
class_ids = sorted(coco.getCatIds())
# All images or a subset?
if class_ids:
    image_ids = []
    for id in class_ids:
        image_ids.extend(list(coco.getImgIds(catIds=[id])))
        coco_imgid = coco.getImgIds(catIds=[id])
        break
    # Remove duplicates
    image_ids = list(set(image_ids))
else:
    # All images
    image_ids = list(coco.imgs.keys())
class_ids = sorted(coco.getCatIds())
print(class_ids)
for id in class_ids:
    label = coco.loadCats(id)[0]["name"]
    print(label)


for i in image_ids:
    path = coco.imgs[i]['file_name']
    print("path: ", path)
    width=coco.imgs[i]["width"]
    print("width: ", width)
    height=coco.imgs[i]["height"]
    print("height: ", height)
    annotations= coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None))
    print("annotations: ", annotations)
    break






























