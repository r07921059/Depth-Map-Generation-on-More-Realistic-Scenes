from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess 
from models import stackhourglass
import cv2
from util import writePFM
from PIL import Image
# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--leftimg', default= None,
                    help='left image')
parser.add_argument('--rightimg', default= None,
                    help='right image')   
parser.add_argument('--isgray', default=False, action='store_true',
                    help='if image is grayscale')                                       
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--output', type=str, default='disparity.pfm',
                    help='output pfm file name')   

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass.PSMNet(args.maxdisp)
elif args.model == 'basic':
    model = basic.PSMNet(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()


real = True

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
    model.eval()

    if args.cuda:
       imgL = torch.FloatTensor(imgL).cuda()
       imgR = torch.FloatTensor(imgR).cuda()     

    imgL, imgR= Variable(imgL), Variable(imgR)

    with torch.no_grad():
        disp = model(imgL,imgR)

    disp = torch.squeeze(disp)
    #disp = disp*1.17
    pred_disp = disp.data.cpu().numpy()

    return pred_disp


def main():
    processed = preprocess.get_transform(augment=False)
    padsize = 64
    if args.isgray:
        imgL_o = cv2.cvtColor(cv2.imread(args.leftimg,0), cv2.COLOR_GRAY2RGB)
        imgR_o = cv2.cvtColor(cv2.imread(args.rightimg,0), cv2.COLOR_GRAY2RGB)
    else:
        # imgL_o = cv2.cvtColor(cv2.imread(args.leftimg), cv2.COLOR_BGR2RGB)
        # imgR_o = cv2.cvtColor(cv2.imread(args.rightimg), cv2.COLOR_BGR2RGB)
        imgL_o = skimage.io.imread(args.leftimg)
        imgR_o = skimage.io.imread(args.rightimg)
        # imgL_o = Image.open(args.leftimg)
        # imgR_o = Image.open(args.rightimg)
    if real:
        imgL_o = np.pad(imgL_o,((0,0),(padsize,0),(0,0)))
        imgR_o = np.pad(imgR_o,((0,0),(0,padsize),(0,0)))
    imgL = processed(imgL_o).numpy()
    imgR = processed(imgR_o).numpy()
    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

    # pad to width and hight to 16 times
    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16       
        top_pad = (times+1)*16 -imgL.shape[2]
    else:
        top_pad = 0
    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3]//16                       
        left_pad = (times+1)*16-imgL.shape[3]
    else:
        left_pad = 0     
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

    start_time = time.time()
    pred_disp = test(imgL,imgR)


    print('time = %.2f' %(time.time() - start_time))
    if top_pad !=0 or left_pad != 0:
        img = pred_disp[top_pad:,:]
    else:
        img = pred_disp

    img = img[:,padsize:]
    writePFM(args.output, img.astype(np.float32))

if __name__ == '__main__':
    main()






