import os
import numpy as np
import argparse
import cv2
import time
from util import writePFM
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
from models import stackhourglass, classifier
from PIL import Image
from torchvision import transforms


def imgPreprocess(Il, model):
    
    img = Image.open(Il).convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = transform(img)
    img = img.unsqueeze(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda")

    if use_cuda:
        img = img.cuda()
    
    with torch.no_grad():
        model.eval()
        pred = model(img)
        pred = pred.argmax(dim=1, keepdim=True).cpu().item()
    
    return pred




# You can modify the function interface as you like
def computeDisp(Il, Ir, model, label, real):
    
    # h, w, ch = Il.shape
    # disp = np.zeros((h, w), dtype=np.int32)
    use_cuda = torch.cuda.is_available()
    maxdisp = 192
    padsize = 64
    torch.manual_seed(1)
    if use_cuda:
        torch.cuda.manual_seed(1)

    processed = preprocess.get_transform(augment=False)
    imgL_o = cv2.cvtColor(cv2.imread(Il, 0), cv2.COLOR_GRAY2RGB)
    imgR_o = cv2.cvtColor(cv2.imread(Ir, 0), cv2.COLOR_GRAY2RGB)
    h, w, c = imgL_o.shape
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

    model.eval()

    if use_cuda:
       imgL = torch.FloatTensor(imgL).cuda()
       imgR = torch.FloatTensor(imgR).cuda()     

    imgL, imgR= Variable(imgL), Variable(imgR)

    with torch.no_grad():
        disp = model(imgL,imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    if top_pad !=0 or left_pad != 0:
        img = pred_disp[top_pad:,:]
    else:
        img = pred_disp

    if real:
        img = img[:,padsize+16:]
        img = cv2.resize(img,(w,h))
        
    img = img.astype(np.float32)

    return img


def main():
    
    parser = argparse.ArgumentParser(description='Disparity Estimation')
    parser.add_argument('--inputDir', default='./data', type=str, help='input data folder')
    parser.add_argument('--outputDir', default='./output', type=str, help='output folder')
    parser.add_argument('--dispModel', default='./pretrained_model_KITTI2012.tar', type=str, help='disparity model')
    parser.add_argument('--classifier', default='./classifier.pth', type=str, help='disparity model')
    args = parser.parse_args()
    
    # check dir
    out_syn_path = os.path.join(args.outputDir, "Synthetic")
    out_real_path = os.path.join(args.outputDir, "Real")
    os.makedirs(out_syn_path, exist_ok=True)
    os.makedirs(out_real_path, exist_ok=True)

    # collect images
    in_syn_path = os.path.join(args.inputDir, "Synthetic")
    in_real_path = os.path.join(args.inputDir, "Real")
    images_fn = []
    
    # syn : 0 ~ 9
    for i in range(10):
        l_file_path = os.path.join(in_syn_path, 'TL{}.png'.format(i))
        r_file_path = os.path.join(in_syn_path, 'TR{}.png'.format(i))
        images_fn.append((l_file_path, r_file_path))
    
    # real : 10 ~ 19
    for i in range(10):
        l_file_path = os.path.join(in_real_path, 'TL{}.bmp'.format(i))
        r_file_path = os.path.join(in_real_path, 'TR{}.bmp'.format(i))
        images_fn.append((l_file_path, r_file_path))
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda")
    
    # load classifier model
    tic = time.time()
    classifier_model = classifier.ImageClassifier()
    classifier_model.load_state_dict(torch.load(args.classifier))
    if use_cuda:
        classifier_model.cuda()
    
    toc = time.time()
    print('load classifier model time: %f sec.' % (toc - tic))
    
    # load PSMNet model
    tic = time.time()
    maxdisp = 192
    disp_model = stackhourglass.PSMNet(maxdisp)
    disp_model = nn.DataParallel(disp_model, device_ids=[0])
    
    if use_cuda:
        disp_model.cuda()
    state_dict = torch.load(args.dispModel)
    disp_model.load_state_dict(state_dict['state_dict'])
    toc = time.time()
    print('load PSMNet model time: %f sec.' % (toc - tic))
    
    # main algo.
    suffix = '.pfm'
    for idx, (l_fn, r_fn) in enumerate(images_fn):
        
        
        tic = time.time()
        label = imgPreprocess(l_fn, classifier_model)
        isReverse = True if label >= 10 else False
        disp = computeDisp(l_fn, r_fn, disp_model, label, isReverse)
        toc = time.time()
        print('process ' + l_fn  + ' time: %f sec.' % (toc - tic))
        
        basefn = os.path.basename(l_fn)
        base = os.path.splitext(basefn)[0]
        
        # syn : 0 ~ 9, real : 10 ~ 19
        pfm_fn = os.path.join(out_syn_path, base + suffix) if idx < 10 else os.path.join(out_real_path, base + suffix)
        writePFM(pfm_fn, disp)
    
    

    

if __name__ == '__main__':
    main()
