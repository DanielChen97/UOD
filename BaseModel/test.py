import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from scipy import misc
from model.ResNet_models import Pred_endecoder
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=480, help='testing size')
parser.add_argument('--feat_channel', type=int, default=256, help='reduced channel of saliency feat')
parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension')

opt = parser.parse_known_args()[0]

dataset_path = '../BaseModel/'

generator = Pred_endecoder(channel=opt.feat_channel)
generator.load_state_dict(torch.load('./models/Model_50_gen.pth'))

generator.cuda()
generator.eval()

test_datasets = ['DUT-OMRON', 'ECSSD', 'DUTS']

for dataset in test_datasets:
    save_path = './results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/img/'
    test_loader = test_dataset(image_root, opt.testsize)
    print(test_loader)
    for i in range(test_loader.size):
        print(i)
        image, HH, WW, name = test_loader.load_data()

        image = image.cuda()
        generator_pred = generator.forward(image)[0]

        res = generator_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res)
