# Abstract 
In this project, i implement the task of bounding box detection for wheat by using Mask_RCNN. The trained model will detect all occurences of wheat inside a single image and then generate bounding boxes for all of them. Notice that the purpose of this project is not to generate masks for detected wheat. 

# Data preparation 
If you intend to train on your custom dataset, please prepare data with the same format as the folder "data" inside this project. It has two subfolders "train" and "val", and a file label.csv that contains information about classes for detection. Folder "data/train/imgs" contains images used for training, and file "data/annotations/annotations.csv" contain bounding box annotations according to these images. Similarly for folder "data/val". 

# Installation 
- tensorflow==1.15.0
- keras==2.0.8

Other packages can be installed by running the command "pip install -r requirements.txt".

# Train 
To train from scratch, please run this command:
"python samples/wheat/train.py train --dataset=data --model=' ' --no_loadmodel"

After training completed, weight will be saved in folder "logs".

If you have trained at least once and want to use trained weight for continue training, please run the command of format 
"python samples/wheat/train.py train --dataset=data --model=path_to_weight --loadmodel".

Please notice that inside class "CoCoConfig" of the file samples/wheat/train.py, we need to set NUM_CLASSES equal to 1 + number of classes, where 1 counts for the background class. For example, this project only detect one class "wheat", so i will set NUM_CLASSES = 1 + 1.

# Test
To test, please run command of format 

python samples/wheat/train.py evaluate --dataset=data --model=weight_path --loadmodel

For example:  

python samples/wheat/train.py evaluate --dataset=data --model=logs/coco20210113T1106/mask_rcnn_coco_0210.h5 --loadmodel

# Reference
The code from this project is based on https://github.com/matterport/Mask_RCNN . For any issue during installation or training, please refer to this link so that you can find more helps from the community.






