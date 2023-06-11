from logger import *
from dataset_sp import LULCDataset
from configs import config_factory
from importlib import import_module
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import sys
import logging
import numpy as np
from tqdm import tqdm
import argparse
from PIL import Image
import scipy, cv2
from evaluate import MscEval

np.set_printoptions(threshold=np.inf)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset',default='gid',choices=['gid','deepglobe'])
    parse.add_argument('--Fold',default=5,type=int)
    parse.add_argument('--Path',default='./res')
    parse.add_argument('--PthFile',default='best')
    parse.add_argument('--Backbone',default='Resnet101')
    parse.add_argument('--ModelFile',default='models.SPSNet_gid_feat')
    parse.add_argument('--Root',default='./data/')
    parse.add_argument('--Validset',default='valid.lst')
    parse.add_argument('--pngPath',default='./classRes/')
    parse.add_argument('--HFTrans',action='store_true')
    parse.add_argument('--VFTrans',action='store_true')
    parse.add_argument('--RRTrans',action='store_true')
    parse.add_argument('--RSTrans',action='store_true')
    parse.add_argument('--RCTrans',action='store_true')
    parse.add_argument('--ClassWeight',action='store_true')
    parse.add_argument('--Background',action='store_true')
    parse.add_argument('--Savepng',action='store_true')
    parse.add_argument('--MultiScale',action='store_true')
    parse.add_argument('--Crop',action='store_true')
    parse.add_argument('--OverlapWeight',action='store_true')
    parse.add_argument('--TTA',action='store_true')
    return parse.parse_args()

def evaluate():
    # setup
    args = parse_args()
    if args.dataset == 'gid':
        cfg = config_factory['config_gid']
    else:
        cfg = config_factory['config_deepglobe']
    if args.Background:
        cfg.n_classes = cfg.n_classes+1
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger()

    logger.info('setup and restore model')
    SPSNet = import_module(args.ModelFile)
    foldCount = args.Fold
    for fold in range(foldCount):
        ## model
        net = SPSNet.SPSNet(cfg, args)
        save_pth = os.path.join(args.Path, args.PthFile+'_'+str(fold)+'.pth')
        net_model = torch.load(save_pth)
        
        for k,v in list(net_model.items()):
            net_model[k[7:]] = net_model.pop(k)
        net.load_state_dict(net_model)
        net.eval()
        net = nn.DataParallel(net).cuda()

        validList = 'val'+str(fold)+'.lst'
        validIndexes = open(os.path.join(args.Path,validList)).readlines()
        validIndexes = np.array(validIndexes,dtype=np.int)

        # evaluator
        logger.info('compute the mIOU')
        evaluator = MscEval(cfg, args, validIndexes)
        val_loss_avg, mIOU, IoU_array, hist = evaluator(net)
        logger.info('Fold:{} mIoU: {:.6f}'.format(fold, mIOU))
        logger.info(IoU_array)



if __name__ == "__main__":
    evaluate()
