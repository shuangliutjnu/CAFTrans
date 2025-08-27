import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from model.FCN.FCN import FCN8s, VGGNet
from model.U_net.unet_model import UNet
from model.segcloud.segcloud import segcloud
from model.cloudsegnet.cloudsegnet import cloudsegnet
from model.PSP.pspnet import PSPNet
from model.FLA.FLA import FLANet
from model.Transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.Transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.Unetformer.UNetFormer import UNetFormer
from model.DCswin.DCSwin import dcswin_small
#from model.Transcloudseg.vit_seg_modeling import VisionTransformer as ViT_seg1
#from model.Transcloudseg.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg1
from model.Intransformer.config import network_config
from model.Intransformer.encoder import Transformer
import model.UNeXt.unext_archs as ua
from model.UNeXt.unext_archs import UNext as unext


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from trainer import trainer_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', default='/home/katie/datasets/5000_labels_6/', type=str, help='Location of dataset')   #/home/xjj/code-pycharm/Cloud/cloud segmentation/dataset/5000/
parser.add_argument('--save_log', type=str, default='./log_bs1_0.8_new_multisoft6', help='Location of train log')                         #/home/xjj/Datasets/5500/    /home/xjj/Datasets/5500_g128/
parser.add_argument('--n_gpu', type=int, default=1, help='the number of GPU for training')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--max_epoch', type=int, default=80, help='maximum epoch number to train')
parser.add_argument('--max_iteration', type=int, default=30000, help='maximum iteration number to train')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--model_saved_mode', type=str, default='total model', help='choose mode for saving model')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--num_classes', type=int, default=3, help='output channel of network')                     ###### 2 ,3
parser.add_argument('--model_name', type=str, default='CAFTans', help='network for training') #FCN ,U-net ,segcloud ,cloudsegnet ,PSP, unetformer
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')        #
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=80, help='maximum epoch number to train')
parser.add_argument('--save_path', type=str, default='/home/katie/cloud_results/result_6_bs1_0.8_new_multisoft6', help='Location of model is saved ')
parser.add_argument('--model_save', type=bool, default=True, help='input patch size of network input')
parser.add_argument('--base_size', type=int, default=512)
parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')

args = parser.parse_args()

if __name__ == '__main__':
    torch.cuda.set_device(1)
    if not args.deterministic:
        deterministic = False
        benchmark = True
    else:
        deterministic = True
        benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # FCN
    if args.model_name == 'FCN':
        vgg_model = VGGNet()
        model = FCN8s(pretrained_net=vgg_model, n_class=args.num_classes).cuda()
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    elif args.model_name == 'U-net':
        model = UNet(n_channels=3, n_classes=args.num_classes, bilinear=True).cuda()
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    elif args.model_name == 'segcloud':
        model = segcloud(n_channels=3, n_classes=args.num_classes, bilinear=True).cuda()
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    elif args.model_name == 'cloudsegnet':
        model = cloudsegnet(n_channels=3, n_classes=args.num_classes, bilinear=True).cuda() #def __init__(self, n_channels, n_classes, bilinear=True):
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    elif args.model_name == 'FLA':
        model = FLANet(n_classes=args.num_classes).cuda()  #def __init__(self, block, layers, n_classes=19, dilated=True, deep_stem=True,zero_init_residual=False, norm_layer=SyncBatchNorm):
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    elif args.model_name == 'PSP':
        # model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=args.num_classes, zoom_factor=8, use_ppm=False, pretrained=True).cuda()
        model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=args.num_classes, zoom_factor=8, use_ppm=False, pretrained=True).cuda()
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    elif args.model_name == 'unetformer':
        model = UNetFormer(num_classes=3).cuda()
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    elif args.model_name == 'Transunet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16'] #R50-ViT-B_16
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=3).cuda()
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    elif args.model_name == 'CAFTans':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16'] #R50-ViT-B_16
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=3).cuda()
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    elif args.model_name == 'DCswin':
        model = dcswin_small(num_classes=3).cuda()
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    # elif args.model_name == 'Transcloudseg':
    #     config_vit1 = CONFIGS_ViT_seg1['R50-ViT-B_16']-----------------------------------------------------------------------------------------------------------------------------
    #     config_vit1.n_classes = args.num_classes
    #     config_vit1.n_skip = 3
    #     config_vit1.patches.grid = (
    #     int(args.img_size / 16), int(args.img_size / 16))
    #     model = ViT_seg1(config_vit1, img_size=args.img_size, num_classes=args.num_classes).cuda()
    #     if args.n_gpu > 1:
    #         model = nn.DataParallel(model)
    elif args.model_name == 'Intransformer':
        config = network_config()
        model = Transformer(config).cuda()
        if args.n_gpu > 1:
            model = nn.DataParallel(model)
    elif args.model_name == 'unext':
        model = ua.__dict__['UNext'](args.num_classes,
                                            3,
                                            False).cuda()
        if args.n_gpu > 1:
            model = nn.DataParallel(model)

    args.save_log = os.path.join(args.save_log, args.model_name)
    if not os.path.exists(args.save_log):
        os.makedirs(args.save_log)
    trainer_dataset(args=args, model=model)








