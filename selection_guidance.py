import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import cv2
from skimage.segmentation import relabel_sequential
from skimage.measure import label as measure_label
import os

#######################################################
################## Model and Device ###################
#######################################################
# import the model from hub
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

# device infor
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# model paras
for p in model.parameters():
        p.requires_grad = False
model.eval()
model.to(device)

#######################################################
#################### Image Load #######################
#######################################################
# load image
images_path = "datasets/MSRA-B/img/"
rbd_seg_path = "datasets/RBD_MSRAB/"
images_opt_seg_path = "output/seg_after_change/"
image_type_jpg = ".jpg"
image_type_png = ".png"

# get all images name
images_name = os.listdir(images_path)
if ".ipynb_checkpoints" in images_name:
    images_name.remove(".ipynb_checkpoints")
images_name.sort()
images_num = len(images_name)
print(images_num)

# transform
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

#######################################################
################### Bounding Box ######################
#######################################################
def get_largest_cc_box(mask: np.array):
    labels = measure_label(mask)  # get connected components
    largest_cc_index = np.argmax(np.bincount(labels.flat)[1:]) + 1
    mask = np.where(labels == largest_cc_index)
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1
    return [xmin, ymin, xmax, ymax]

def get_bbox_from_patch_mask(patch_mask, image):

    # Sizing
    H = image.shape[0]
    W = image.shape[1]
    T = patch_mask.numel()
    H_lr = 0
    W_lr = 0
    if (H // 8) * (W // 8) == T:
        P, H_lr, W_lr = (8, H // 8, W // 8)
    elif (H // 16) * (W // 16) == T:
        P, H_lr, W_lr = (16, H // 16, W // 16)
    elif 4 * (H // 16) * (W // 16) == T:
        P, H_lr, W_lr = (8, 2 * (H // 16), 2 * (W // 16))
    elif 16 * (H // 32) * (W // 32) == T:
        P, H_lr, W_lr = (8, 4 * (H // 32), 4 * (W // 32))

    # Create patch mask
    patch_mask = patch_mask.reshape(H_lr, W_lr).cpu().numpy()
    
    # Possibly reverse mask
    # print(np.mean(patch_mask).item())
    if 0.5 < np.mean(patch_mask).item() < 1.0:
        patch_mask = (1 - patch_mask).astype(np.uint8)
    elif np.sum(patch_mask).item() == 0:  # nothing detected at all, so cover the entire image
        patch_mask = (1 - patch_mask).astype(np.uint8) 
    
    # Get the box corresponding to the largest connected component of the first eigenvector
    xmin, ymin, xmax, ymax = get_largest_cc_box(patch_mask)
    # pred = [xmin, ymin, xmax, ymax]

    # Rescale to image size
    r_xmin, r_xmax = P * xmin, P * xmax
    r_ymin, r_ymax = P * ymin, P * ymax

    # Prediction bounding box
    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    # Check not out of image size (used when padding)
    pred[2] = min(pred[2], W)
    pred[3] = min(pred[3], H)

    return np.asarray(pred)

#######################################################
################### Segmentation ######################
#######################################################
def get_every_seg_bi_img_for_each_attention_map(attention, thres, image):
    
    patch_mask = (attention >= thres)
    pred = get_bbox_from_patch_mask(patch_mask, image)
    
    mask = np.zeros(image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)    
    fgdModel = np.zeros((1,65),np.float64)    
    rect = [0,0,0,0]                         

    rect_copy = (pred[0], pred[1], pred[2]-pred[0], pred[3]-pred[1])
    if rect_copy == (0,0,image.shape[1], image.shape[0]):
        rect_copy = (1,1,image.shape[1]-2, image.shape[0]-2)
        
    cv2.grabCut(image, mask,rect_copy, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    white_image = np.zeros_like(image, np.uint8) + 255
    seg_bi_image = white_image*mask2[:,:,np.newaxis]
    
    return seg_bi_image

def bi_image_three_channels_to_one(image):
    
    return image[:,:,0]

def get_acc(seg_img_1, seg_img_2):
    
    seg_img_1_bol = seg_img_1.astype(np.bool)
    seg_img_2_bol = seg_img_2.astype(np.bool)
    
    out = seg_img_1_bol * seg_img_2_bol + ~seg_img_1_bol * ~seg_img_2_bol
    correct_pixels = np.sum(out == True)
    
    return correct_pixels/(seg_img_1.shape[0]*seg_img_1.shape[1])

seg_path = "output/seg_after_change/"
RBD_MSRAB_path = "datasets/RBD_MSRAB/"

images_name = os.listdir(seg_path)
if ".ipynb_checkpoints" in images_name:
    images_name.remove(".ipynb_checkpoints")
images_name.sort()
images_num = len(images_name)
print(images_num)


#######################################################
############### Selection Guidance ####################
#######################################################
patch_size = 8

for i in range(images_num):

    image = plt.imread(images_path + images_name[i])
    if len(image.shape) == 2:
        image = cv2.imread(images_path + images_name[i])
    
    image_num_in_name = images_name[i].split('.jpg')[0]

    img = transformer(image)
    
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_selfattention(img.to(device)) 

    nh = attentions.shape[1] # number of head 6

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1) 

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions_mean = attentions.mean(-3)
    
    thres = attentions_mean.max() * 0.07
    
    seg_bi_image_0 = get_every_seg_bi_img_for_each_attention_map(attentions[0], thres, image)
    seg_bi_image_0_one_channel = bi_image_three_channels_to_one(seg_bi_image_0)
    
    seg_bi_image_1 = get_every_seg_bi_img_for_each_attention_map(attentions[1], thres, image)
    seg_bi_image_1_one_channel = bi_image_three_channels_to_one(seg_bi_image_1)
    
    seg_bi_image_2 = get_every_seg_bi_img_for_each_attention_map(attentions[2], thres, image)
    seg_bi_image_2_one_channel = bi_image_three_channels_to_one(seg_bi_image_2)
    
    seg_bi_image_3 = get_every_seg_bi_img_for_each_attention_map(attentions[3], thres, image)
    seg_bi_image_3_one_channel = bi_image_three_channels_to_one(seg_bi_image_3)
    
    seg_bi_image_4 = get_every_seg_bi_img_for_each_attention_map(attentions[4], thres, image)
    seg_bi_image_4_one_channel = bi_image_three_channels_to_one(seg_bi_image_4)
    
    seg_bi_image_5 = get_every_seg_bi_img_for_each_attention_map(attentions[5], thres, image)
    seg_bi_image_5_one_channel = bi_image_three_channels_to_one(seg_bi_image_5)
    
    seg_bi_image_mean = get_every_seg_bi_img_for_each_attention_map(attentions_mean, thres, image)
    seg_bi_image_mean_one_channel = bi_image_three_channels_to_one(seg_bi_image_mean)
    
    rbd_seg_image = cv2.imread(rbd_seg_path + image_num_in_name + image_type_png, 0)
    rbd_seg_image[rbd_seg_image >= 128] = 255
    rbd_seg_image[rbd_seg_image < 128] = 0
    
    seg_bi_images_list = [seg_bi_image_0_one_channel, seg_bi_image_1_one_channel, seg_bi_image_2_one_channel, seg_bi_image_3_one_channel, seg_bi_image_4_one_channel, seg_bi_image_5_one_channel, seg_bi_image_mean_one_channel]
    
    acc_0 = get_acc(seg_bi_image_0_one_channel, rbd_seg_image)
    acc_1 = get_acc(seg_bi_image_1_one_channel, rbd_seg_image)
    acc_2 = get_acc(seg_bi_image_2_one_channel, rbd_seg_image)
    acc_3 = get_acc(seg_bi_image_3_one_channel, rbd_seg_image)
    acc_4 = get_acc(seg_bi_image_4_one_channel, rbd_seg_image)
    acc_5 = get_acc(seg_bi_image_5_one_channel, rbd_seg_image)
    acc_mean = get_acc(seg_bi_image_mean_one_channel, rbd_seg_image)
    
    acc_list = [acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_mean]
    
    index = np.argmax(acc_list)
    
    if ((seg_bi_images_list[index] == np.zeros_like(seg_bi_images_list[index])).all()):
        plt.imsave(fname=images_opt_seg_path+image_num_in_name+".jpg", arr=rbd_seg_image, format='jpg')
    else:
        plt.imsave(fname=images_opt_seg_path+image_num_in_name+".jpg", arr=seg_bi_images_list[index], format='jpg')

    print(str(i+1) + ". Successfully saved "+image_num_in_name+".jpg")