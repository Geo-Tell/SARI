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
import cv2
from scipy import ndimage
np.set_printoptions(threshold=np.inf)
labels_dict_gid = [{'name':'built-up','color':(255,0,0)},
        {'name':'farmland','color':(0,255,0)},
        {'name':'forest','color':(128,255,255)},
        {'name':'meadow','color':(255,255,0)},
        {'name':'water','color':(0,0,255)},
        {'name':'others','color':(255,255,255)}
        ]

labels_dict_deepglobe = [{'name':'urban land','color':(0,255,255)},
        {'name':'agriculture land','color':(255,255,0)},
        {'name':'rangeland','color':(255,0,255)},
        {'name':'forest land','color':(0,255,0)},
        {'name':'water','color':(0,0,255)},
        {'name':'barren land','color':(255,255,255)},
        {'name':'others','color':(0,0,0)}]
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset',default='gid',choices=['gid','deepglobe'])
    parse.add_argument('--Path',default='./res')
    parse.add_argument('--PthFile',default='best.pth')
    parse.add_argument('--Backbone',default='Resnet101')
    parse.add_argument('--ModelFile',default='models.SPSNet_deepglobe_feat')
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

class MscEval(object):
    def __init__(self, cfg, bashArgs, indexes=None, *args, **kwargs):
        self.cfg = cfg
        self.args = bashArgs
        self.val_ds = LULCDataset(cfg, bashArgs, bashArgs.Validset, 'val', indexes)
        self.val_dl = DataLoader(self.val_ds,
                    batch_size = 1,
                    shuffle = False,
                    num_workers = 2,
                    pin_memory = True,
                    drop_last = False)

    def __call__(self, net):
        ## evaluate
        if not self.args.MultiScale:
            self.cfg.eval_scales = (1,)
        net.eval()
        val_loss_avg=[]
        n_classes = self.cfg.n_classes
        ignore_label = self.cfg.ignore_label

        val_criteria = torch.nn.CrossEntropyLoss(ignore_index=self.cfg.ignore_label).cuda()
        hist = torch.zeros(n_classes, n_classes).cuda()
        ious = np.array([])
        vflip_option = [False]
        hflip_option = [False]
        angle_option = [0]
        if self.args.TTA:
            vflip_option = [False, True]
            hflip_option = [False, True]
            angle_option = [0, 1]
        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.val_dl), 0):
                images, labels, _, fn = batch
                labels = labels.squeeze(1).cuda()
                N, H, W = labels.shape
                probs = torch.zeros((N, n_classes, H, W)).cuda()
                probs.requires_grad = False
                for vflip in vflip_option:
                    for hflip in hflip_option:
                        images_tmp = images.numpy()[0].transpose((1,2,0)).copy()
                        if vflip:
                            images_tmp = images_tmp[::-1,:,:]
                        if hflip:
                            images_tmp = images_tmp[:,::-1,:]
                        for angle in angle_option:
                            images_tmp = np.rot90(images_tmp, angle, (0, 1))
                            for scale in self.cfg.eval_scales:
                                w, h = int(W * scale + 0.5), int(H * scale + 0.5)
                                new_img = cv2.resize(images_tmp,(w, h))
                                new_img = new_img.transpose((2, 0, 1))
                                new_img = np.expand_dims(new_img, axis=0)
                                new_img = torch.from_numpy(new_img)
                                if not self.args.Crop:
                                    _, logits = net(new_img)
                                else:
                                    crop_size = self.cfg.crop_size[0]
                                    if h <= crop_size and w <= crop_size:
                                        _, logits = net(new_img)
                                    else:
                                        overlap = int(crop_size / 2)
                                        H_count = np.int(np.ceil((h - overlap) / (crop_size - overlap)))
                                        W_count = np.int(np.ceil((w - overlap) / (crop_size - overlap)))
                                        logits = torch.zeros((N, n_classes, h, w)).cuda()
                                        count = torch.zeros((1,1, h, w)).cuda()
                                        interval = overlap - ((H_count+1)*overlap-h)/(H_count-1)
                                        if self.args.OverlapWeight:
                                            patch_weights = np.ones((crop_size + 2, crop_size + 2),
                                                                    dtype=np.float)
                                            patch_weights[0, :] = 0
                                            patch_weights[-1, :] = 0
                                            patch_weights[:, 0] = 0
                                            patch_weights[:, -1] = 0
                                            patch_weights = ndimage.distance_transform_edt(patch_weights)
                                            patch_weights = patch_weights[1:-1, 1:-1]
                                            patch_weights = torch.tensor(patch_weights).float().cuda()
                                            patch_weights = patch_weights[None, None, :, :]  # Adding batch and channels dims
                                        else:
                                            patch_weights = np.ones((crop_size, crop_size),
                                                                    dtype=np.float)
                                            patch_weights = torch.tensor(patch_weights).float().cuda()
                                        for Hi in range(H_count):
                                            for Wj in range(W_count):
                                                h0 = int(Hi * interval)
                                                w0 = int(Wj * interval)
                                                h1 = h0 + crop_size
                                                w1 = w0 + crop_size
                                                im_part = new_img[:,:,h0:h1,w0:w1]
                                                _, logits_part = net(im_part)
                                                logits[:,:,h0:h1,w0:w1] += logits_part * patch_weights
                                                count[:,:,h0:h1,w0:w1] += patch_weights
                                        logits = logits / count
                                logits = F.interpolate(logits,(H,W),mode='bilinear',align_corners=True)
                                prob = F.softmax(logits.cuda(), 1)
                                prob = torch.rot90(prob, -angle, (2,3))
                                if vflip:
                                    prob = torch.flip(prob,[2])
                                if hflip:
                                    prob = torch.flip(prob,[3])
                                probs += prob
                preds = torch.argmax(probs, dim=1)
                labels_temp = labels.clone()
                loss = val_criteria(logits, labels_temp)
                val_loss_avg.append(loss.item())
                if self.args.Savepng:
                    color_save_path = os.path.join(self.args.Path,self.args.pngPath)
                    if not os.path.exists(color_save_path):
                        os.makedirs(color_save_path)
                    color_preds = preds.clone()
                    color_preds = color_preds.squeeze(0).unsqueeze(-1).repeat(1,1,3).cpu().numpy().astype('uint8')
                    
                    if self.args.dataset == 'gid':
                        for i in range(len(labels_dict_gid)+self.args.Background-1):
                            color_preds[(preds.squeeze(0).cpu().numpy()==i)]=labels_dict_gid[i]['color']
                        color_preds[(labels.squeeze(0).cpu().numpy()==255)] = (0,0,0)
                    else:
                        for i in range(len(labels_dict_deepglobe)+self.args.Background-1):
                            color_preds[(preds.squeeze(0).cpu().numpy()==i)]=labels_dict_deepglobe[i]['color']
                        color_preds[(labels.squeeze(0).cpu().numpy()==255)] = (255,255,255)
                    save_color_img = Image.fromarray(color_preds)
                    save_color_img.save(os.path.join(color_save_path, fn[0]+".png"))

                    color_label = labels.squeeze(0).unsqueeze(-1).repeat(1,1,3).cpu().numpy().astype('uint8')
                    if self.args.dataset == 'gid':
                        for i in range(len(labels_dict_gid)+self.args.Background-1):
                            color_label[(labels.squeeze(0).cpu().numpy()==i)]=labels_dict_gid[i]['color']
                        color_label[(labels.squeeze(0).cpu().numpy()==255)] = (0,0,0)
                    else:
                        for i in range(len(labels_dict_deepglobe)+self.args.Background-1):
                            color_label[(labels.squeeze(0).cpu().numpy()==i)]=labels_dict_deepglobe[i]['color']
                        color_label[(labels.squeeze(0).cpu().numpy()==255)] = (255,255,255)
                    save_color_label = Image.fromarray(color_label)
                    save_color_label.save(os.path.join(color_save_path, fn[0]+"_label.png"))
                keep = labels != ignore_label
                hist += torch.bincount(
                    labels[keep] * n_classes + preds[keep],
                    minlength=n_classes ** 2
                    ).to(torch.float32).view(n_classes, n_classes)
        PA = hist.diag().sum()/hist.sum()
        # print(PA)
        
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        ious_np = ious.cpu().numpy()
        if self.args.Background:
            miou = np.nanmean(ious_np[:-1])
        else:
            miou = np.nanmean(ious_np)
        return sum(val_loss_avg)/len(val_loss_avg), miou.item(), ious, hist

def evaluate():
    ## setup
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

    ## model
    logger.info('setup and restore model')
    SPSNet = import_module(args.ModelFile)
    net = SPSNet.SPSNet(cfg, args)
    save_pth = os.path.join(args.Path, args.PthFile)
    net_model = torch.load(save_pth)
    for k,v in list(net_model.items()):
        net_model[k[7:]] = net_model.pop(k)
    net.load_state_dict(net_model)
    net.cuda()
    net.eval()
    net = nn.DataParallel(net)

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(cfg, args)
    val_loss_avg, mIOU, IoU_array, hist = evaluator(net)
    logger.info('val loss is: {:.6f}, mIOU is: {:.6f}'.format(val_loss_avg, mIOU))
    logger.info(IoU_array)
    logger.info(hist)

if __name__ == "__main__":
    evaluate()
