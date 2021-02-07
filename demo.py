import sys
import os
# Root directory of the project
ROOT_DIR = os.path.abspath("")
sys.path.insert(0, "/home/dat/AI_2D/Detection/Mask_RCNN/samples/coco")

import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config

import coco


# Directory to save logs and trained model

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH =  "logs/coco20210114T0341/mask_rcnn_coco_0090.h5"
# Download COCO trained weights from Releases if needed

# Directory of images to run detection on
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'wheat']

#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
import glob
imgpath_list = glob.glob("data/test/imgs/*.jpg")
image = skimage.io.imread(imgpath_list[1])
print(imgpath_list[1])
# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])















