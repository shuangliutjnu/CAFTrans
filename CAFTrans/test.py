import argparse
import logging
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import torch.nn as nn
from evaluation import test_evaluation, save_image_mask
from model.FCN.FCN import FCN32s, VGGNet
from model.U_net.unet_model import UNet
from model.segcloud.segcloud import segcloud
from model.cloudsegnet.cloudsegnet import cloudsegnet
from model.PSP.pspnet import PSPNet
from model.FLA.FLA import FLANet
# from model.FLJ.FLJ import FLJNet
from model.Unetformer.UNetFormer import UNetFormer
# from model.HRViT.hrvit_resnet50 import HRViT_b3_224
from model.Transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.Transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.Unetformer.UNetFormer import UNetFormer
from model.DCswin.DCSwin import dcswin_small

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', default='/home/katie/datasets/5000_labels_6/', type=str, help='Location of dataset')   #/home/xjj/code-pycharm/Cloud/cloud segmentation/dataset/5000/
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--loaded_model_type', type=str, default='total model', help='load model dict or total model')
parser.add_argument('--n_gpu', type=int, default=1, help='the number of GPU for using test')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--evaluation', type=str, default='test_evaluation', help='test_evaluation or save_image_mask')
parser.add_argument('--save_image_path', type=str, default='/home/katie/image-cloud/', help='Location of save image mask')
parser.add_argument('--load_model', default='/home/katie/cloud_results/result_6_bs1_0.8_new_multisoft6/SFtransformer/epoch25.pth', type=str, #/home/katie/result_5000_labels /home/katie/code/cloud_ly/result/cloudsegnet/epoch100.pth
                   help='Location of model which is loaded')                                                                                     #FCN ,U-net ,segcloud ,cloudsegnet ,PSP
parser.add_argument('--save_log', type=str, default='', help='Location of test log')
parser.add_argument('--model_name', type=str, default='SFtransformer', help='network for training') # Seg2, unetformer
parser.add_argument('--base_size', type=int, default=512)
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
args = parser.parse_args()


if __name__ == '__main__':
    torch.cuda.set_device(1)
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.loaded_model_type == 'total model':
        model = torch.load(args.load_model)
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    else:
        if args.model_name == 'FCN':
            vgg_model = VGGNet()
            model = FCN32s(pretrained_net=vgg_model, n_class=args.num_classes).cuda()
            model = model.load_state_dict(torch.load(args.load_model))
            if args.n_gpu > 1:
                model = nn.DataParallel(model)
        elif args.model_name == 'U-net':
            model = UNet(n_channels=3, n_classes=args.num_classes, bilinear=True).cuda()
            model = model.load_state_dict(torch.load(args.load_model))
            if args.n_gpu > 1:
                model = nn.DataParallel(model)
        elif args.model_name == 'segcloud':
            model = segcloud(n_channels=3, n_classes=args.num_classes, bilinear=True).cuda()
            model = model.load_state_dict(torch.load(args.load_model))
            if args.n_gpu > 1:
                model = nn.DataParallel(model)
        elif args.model_name == 'cloudsegnet':
            model = cloudsegnet(n_channels=3, n_classes=args.num_classes, bilinear=True).cuda()
            model = model.load_state_dict(torch.load(args.load_model))
            if args.n_gpu > 1:
                model = nn.DataParallel(model)
        elif args.model_name == 'FLA':
            model = FLANet(n_classes=args.num_classes).cuda()
            if args.n_gpu > 1:
                model = nn.DataParallel(model)
        elif args.model_name == 'unetformer':
            model = UNetFormer(num_classes=3).cuda()
            if args.n_gpu > 1:
                model = nn.DataParallel(model)
        elif args.model_name == 'PSP':
            model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=args.num_classes, zoom_factor=8,use_ppm=False, pretrained=True).cuda()
            if args.n_gpu > 1:
                model = nn.DataParallel(model)
        elif args.model_name == 'Transunet':
            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']  # R50-ViT-B_16
            model = ViT_seg(config_vit, img_size=args.img_size, num_classes=3).cuda()
            if args.n_gpu > 1:
                model = nn.DataParallel(model)
        elif args.model_name == 'DCswin':
            model = dcswin_small(num_classes=3).cuda()
            model = model.load_state_dict(torch.load(args.load_model))
            if args.n_gpu > 1:
                model = nn.DataParallel(model)



    if args.evaluation == 'test_evaluation':
        test_evaluation(args=args, model=model)
    else:
        save_image_mask(args=args, model=model)
