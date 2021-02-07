import os
import glob

def write_csv(filepath, dict_):
    import csv
    with open(filepath, 'w') as file:
        writer = csv.writer(file)
        for key,value in dict_.items():
            writer.writerow([key,value])

def read_label(list_id, target_img_folder):
    import csv
    import ast
    dict_info = {}
    with open('data/train.csv', 'r') as file:
        reader = csv.reader(file)
        index = 0
        for row in list(reader)[1:]:
            img_id, width, height, bbox, source = row
            if img_id in list_id:
                width = int(width)
                height = int(height)
                bbox = ast.literal_eval(bbox)
                # x,y,w,h
                # segmentation = {'segmentation': save_mask, 'area': area, 'iscrowd': iscrowd, 'image_id': image_id,'bbox': bbox, 'category_id': category_id, 'id': id_box }
                area = width*height
                segmentation = {'area': area, 'iscrowd':0, 'image_id': img_id, 'bbox': bbox, 'category_id': 1,  'id': index}
                index += 1
                # [img_path, img_name, self.width, self.height, [segmentation]]
                img_name = img_id+'.jpg'
                img_path = os.path.join(target_img_folder, img_name)

                if img_id not in dict_info.keys():
                    dict_info[img_id] = [img_path, img_name, width, height, [segmentation]]
                else:
                    dict_info[img_id][4].append(segmentation)
    return dict_info

def transfer_data():
    from random import shuffle
    import shutil
    original_dir = "data/train_"
    destination_traindir = "data/train"
    destination_valdir = "data/val"
    destination_testdir = "data/test"
    list_imgname = os.listdir(original_dir)
    shuffle(list_imgname)
    train_imgname = list_imgname[:1000]
    val_imgname = list_imgname[1000:1200]
    test_imgname = list_imgname[1200:1400]

    for imgname in train_imgname:
        old_imgpath = os.path.join(original_dir, imgname)
        new_imgpath = os.path.join(destination_traindir, 'imgs', imgname)
        shutil.copyfile(old_imgpath,new_imgpath)
    for imgname in val_imgname:
        old_imgpath = os.path.join(original_dir, imgname)
        new_imgpath = os.path.join(destination_valdir, 'imgs', imgname)
        shutil.copyfile(old_imgpath,new_imgpath)
    for imgname in test_imgname:
        old_imgpath = os.path.join(original_dir, imgname)
        new_imgpath = os.path.join(destination_testdir, 'imgs', imgname)
        shutil.copyfile(old_imgpath, new_imgpath)

    list_train_id = [x.split('.')[0] for x in train_imgname]
    list_val_id = [x.split('.')[0] for x in val_imgname]
    list_test_id = [x.split('.')[0] for x in test_imgname]

    train_dictinfo = read_label(list_train_id, os.path.join(destination_traindir, "imgs"))
    val_dictinfo = read_label(list_val_id, os.path.join(destination_valdir, "imgs"))
    test_dictinfo = read_label(list_test_id, os.path.join(destination_testdir, "imgs"))

    write_csv(os.path.join(destination_traindir, 'annotations/annotations.csv'), train_dictinfo)
    write_csv(os.path.join(destination_valdir, 'annotations/annotations.csv'), val_dictinfo)
    write_csv(os.path.join(destination_testdir, 'annotations/annotations.csv'), test_dictinfo)


transfer_data()


























