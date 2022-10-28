import numpy as np
import cv2
from BFS import bilateral_solver_output
import matplotlib.pyplot as plt
import os


img_path = "datasets/DUTS-TE/img/"
seg_path = "BaseModel/results/DUTS-TE/"
save_path = "BaseModel/results/DUTS-TE-BS/"


images_name = os.listdir(img_path)
if ".ipynb_checkpoints" in images_name:
    images_name.remove(".ipynb_checkpoints")
images_name.sort()
images_num = len(images_name)
print(images_num)
print(images_name[0])


for i in range(images_num):
    print(i)
    image_name = images_name[i].split('.jpg')[0]
    
    bip_image = cv2.imread(seg_path+image_name+'.png', 0)
    bip_image[bip_image >= 128] = 255
    bip_image[bip_image < 128] = 0

    output_solver, binary_solver = bilateral_solver_output(img_path+image_name+'.jpg', bip_image, sigma_spatial = 16, sigma_luma = 16, sigma_chroma = 8)
    
    thred = 128
    output_solver[output_solver >= thred] = 255
    output_solver[output_solver < thred] = 0
    
    res = np.zeros((output_solver.shape[0], output_solver.shape[1], 3), dtype="uint8")
    res[:,:,0] = output_solver
    res[:,:,1] = output_solver
    res[:,:,2] = output_solver
    
    plt.imsave(fname=save_path+image_name+'.png', arr=res, format='png')