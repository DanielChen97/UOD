import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


gt_path = "datasets/DUTS-TE/gt/"
seg_path = "BaseModel/results/DUTS-TE-BS/"

images_name = os.listdir(seg_path)
if ".ipynb_checkpoints" in images_name:
    images_name.remove(".ipynb_checkpoints")
images_name.sort()
images_num = len(images_name)
print(images_num)


sum_acc = 0
sum_iou = 0

for i in range(images_num):
    
    img_name = images_name[i]
    print(i)
    
    gt_img = cv2.imread(gt_path + img_name, 0)

    seg_img = cv2.imread(seg_path + img_name, 0)
    seg_img[seg_img >= 128] = 255
    seg_img[seg_img < 128] = 0
    
    # acc
    gt_img_state = gt_img.astype(np.bool)
    seg_img = seg_img.astype(np.bool)
    
    out = gt_img_state * seg_img + ~gt_img_state * ~seg_img

    correct_pixels = np.sum(out == True)
    
    sum_acc = sum_acc + correct_pixels/(gt_img.shape[0]*gt_img.shape[1])
    
    intersection = np.logical_and(gt_img, seg_img) 
    union = np.logical_or(gt_img, seg_img) 
    iou_score = np.sum(intersection) / np.sum(union)
    
    sum_iou = sum_iou + iou_score

ave_acc = sum_acc / images_num
ave_iou = sum_iou / images_num
print("ave acc: ", ave_acc)
print("ave iou: ", ave_iou)