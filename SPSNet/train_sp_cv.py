from logger import *
from dataset_sp import LULCDataset
from evaluate import MscEval
from loss import OhemCELoss, Lvar, Linter, Lintra
from configs import config_factory
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import logging
import time
import argparse
from importlib import import_module
import os.path as osp

torch.backends.cudnn.benchmark = True

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset',default='gid',choices=['gid','deepglobe'])
    parse.add_argument('--resPath',default='./res')
    parse.add_argument('--HFTrans',action='store_true')
    parse.add_argument('--VFTrans',action='store_true')
    parse.add_argument('--RRTrans',action='store_true')
    parse.add_argument('--RSTrans',action='store_true')
    parse.add_argument('--RCTrans',action='store_true')
    parse.add_argument('--Root',default='./data/')
    parse.add_argument('--Trainset',default='train.lst')
    parse.add_argument('--Validset',default='val.lst')
    parse.add_argument('--Epoch', type=int)
    parse.add_argument('--Folds',default=5, type=int)
    parse.add_argument('--NowFold',default=0, type=int)
    parse.add_argument('--Delay', type=int, default=75)
    parse.add_argument('--lrStart1',default=3e-4,type=float)
    parse.add_argument('--lrStart2',default=1e-4,type=float)
    parse.add_argument('--ModelFile',default='models.SPSNet_gid_feat')
    parse.add_argument('--Backbone',default='Resnet101_HDC')
    parse.add_argument('--ClassWeight',action='store_true')
    parse.add_argument('--BestNum', default=0, type=int)
    parse.add_argument('--Background',action='store_true')
    parse.add_argument('--lamda1',default=0.1,type=float)
    parse.add_argument('--lamda2',default=0.1,type=float)
    parse.add_argument('--lamda3',default=50,type=float)
    parse.add_argument('--MultiScale',action='store_true')
    parse.add_argument('--Crop',action='store_true')
    parse.add_argument('--OverlapWeight',action='store_true')
    parse.add_argument('--TTA',action='store_true')
    parse.add_argument('--Savepng',action='store_true')
    parse.add_argument('--Resume',action='store_true')
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    return parse.parse_args()

def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/(max_iters+1))**(power))
    optimizer.param_groups[0]['lr'] = lr


def get_kfold_data(k, length):  
    fold_size = length // k
    index = np.arange(0,length,1).astype(np.int)
    # np.random.shuffle(index)
    train_indexes = []
    valid_indexes = []
    for i in range(k):
        val_start = i * fold_size
        if i != k - 1:
            val_end = (i + 1) * fold_size
            valid_index = index[val_start:val_end]
            train_index = index[0:val_start]
            train_index = np.append(train_index,index[val_end:])
        else:
            valid_index = index[val_start:]
            train_index = index[0:val_start]
        train_indexes.append(train_index)
        valid_indexes.append(valid_index)
    return np.array(train_indexes,dtype=object), np.array(valid_indexes,dtype=object)

def train(verbose=True, **kwargs):
    args = kwargs['args']
    cfg = kwargs['cfg']
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    SPSNet = import_module(args.ModelFile)
    if args.Background:
        cfg.n_classes = cfg.n_classes+1
    if args.local_rank <= 0:
        if not osp.exists(args.resPath): os.makedirs(args.resPath)
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        logfile = 'SPSNet-{}.log'.format(time_str)
        logfile = osp.join(args.resPath, logfile)
        setup_logger(logfile)
        logger = logging.getLogger()
        logger.info('logger setup')

    # dataset
    train_indexes = [[] for _ in range(args.Folds)]
    valid_indexes = [[] for _ in range(args.Folds)]
    train_list = 'train'+str(args.NowFold)+'.lst'
    valid_list = 'val'+str(args.NowFold)+'.lst'
    train_list_file = os.path.join(args.resPath,train_list)
    valid_list_file = os.path.join(args.resPath,valid_list)

    if args.Resume and os.path.exists(train_list_file):
        trainIndexes = open(train_list_file).readlines()
        trainIndexes = np.array(trainIndexes,dtype=np.int)
        validIndexes = open(train_list_file).readlines()
        validIndexes = np.array(validIndexes,dtype=np.int)
        train_indexes[args.NowFold]=trainIndexes
        valid_indexes[args.NowFold]=validIndexes
    else:
        ds1 = LULCDataset(cfg, args, args.Trainset, 'train')
        train_indexes, valid_indexes = get_kfold_data(args.Folds,len(ds1))

        train_file = open(train_list_file,'w')
        valid_file = open(valid_list_file,'w')
        for i in train_indexes[args.NowFold]:
            train_file.write(str(i)+'\n')
        for i in valid_indexes[args.NowFold]:
            valid_file.write(str(i)+'\n')
        train_file.close()
        valid_file.close()

    if args.local_rank <= 0:
        logger.info('fold:'+str(args.NowFold))
    ds1 = LULCDataset(cfg, args, args.Trainset, 'train', train_indexes[args.NowFold])
    ds2 = LULCDataset(cfg, args, args.Trainset, 'train_sp', train_indexes[args.NowFold])
    sampler1 = torch.utils.data.distributed.DistributedSampler(ds1)
    sampler2 = torch.utils.data.distributed.DistributedSampler(ds2)
    dl1 = DataLoader(ds1,
                    batch_size = cfg.ims_per_gpu*2,
                    shuffle = False,
                    sampler=sampler1,
                    num_workers = cfg.n_workers,
                    pin_memory = True,
                    drop_last = True)
    dl2 = DataLoader(ds2,
                    batch_size = cfg.ims_per_gpu,
                    shuffle = False,
                    sampler=sampler2,
                    num_workers = cfg.n_workers,
                    pin_memory = True,
                    drop_last = True)
    if args.local_rank <= 0:
        logger.info('dataset ditributed')

    ## model
    net = SPSNet.SPSNet(cfg, args)
    net = net.cuda(args.local_rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
    if args.local_rank <= 0:
        logger.info('net initialized')

    ## optimizer
    if hasattr(net, 'module'):
        wd_params, non_wd_params = net.module.get_params()
    else:
        wd_params, non_wd_params = net.get_params()
    params_list = [{'params': wd_params,},
                {'params': non_wd_params, 'weight_decay': cfg.weight_decay}]
    optimizer = torch.optim.AdamW(params_list,lr=args.lrStart1,weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0)
    if args.local_rank <= 0:
        logger.info('optimizer done')

    epoch_iters1 = np.int(ds1.__len__() / 
                        cfg.ims_per_gpu / 2 / torch.cuda.device_count())
    epoch_iters2 = np.int(ds2.__len__() / 
                        cfg.ims_per_gpu / torch.cuda.device_count())
    best_mIoU = 0
    last_epoch = 0
    omit_list = -torch.ones(1).long()
    if args.local_rank <= 0:
        tensorboard_log_dir = os.path.join(args.resPath, 'tensorboard_'+str(args.NowFold)+'_'+time_str)
    if args.Resume:
        #writer
        for root, dirs, files in os.walk(args.resPath):
            dirs.sort(reverse = True)
            for tensor_dirs in dirs:
                if 'tensorboard_'+str(args.NowFold) in tensor_dirs:
                    tensorboard_log_dir = os.path.join(args.resPath,tensor_dirs)
                    break
        model_state_file = os.path.join(args.resPath,
                                        'checkpoint_'+str(args.NowFold)+'.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            omit_list = checkpoint['omit_list'].cuda()
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if last_epoch < args.Delay:
                for i in range(last_epoch):
                    scheduler.step()
            if args.local_rank <= 0:
                logger.info("=> loaded checkpoint (epoch {})"
                            .format(checkpoint['epoch']))
            del checkpoint
            torch.distributed.barrier()
    if args.local_rank <= 0:
        if not osp.exists(tensorboard_log_dir): os.makedirs(tensorboard_log_dir)
    
    if args.local_rank <= 0:
        writer_dict = {
            'writer': SummaryWriter(tensorboard_log_dir),
            'train_global_steps': last_epoch,
            'valid_global_steps': last_epoch,
        }

    end_epoch = args.Epoch
    n_min = 200*200
    criteria = OhemCELoss(thresh=cfg.ohem_thresh, n_min=n_min, ignore_lb=cfg.ignore_label, weight=ds1.class_weights).cuda()
    criteria1 = Lvar(n_classes=cfg.n_classes, ignore_lb=cfg.ignore_label).cuda()
    criteria2 = Linter(n_classes=cfg.n_classes, ignore_lb=cfg.ignore_label).cuda()
    criteria3 = Lintra(n_classes=cfg.n_classes, ignore_lb=cfg.ignore_label).cuda()
    
    if args.local_rank <= 0:
        logger.info('train start')
    for epoch in range(last_epoch, args.Delay):
        net.train()
        sampler1.set_epoch(epoch)
        loss_avg = []
        loss1_avg = []
        if args.local_rank <= 0:
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
        st = time.time()
        for i_iter, batch in enumerate(dl1,0):
            images, labels, _, _ = batch
            images = images.cuda()
            labels = labels.cuda()
            labels = torch.squeeze(labels, 1)
            optimizer.zero_grad()
            logits = net(images, feat_map=False)
            loss1 = criteria(logits, labels)
            loss = loss1
            loss.backward()
            optimizer.step()
            
            loss_avg.append(loss.item())
            loss1_avg.append(loss1.item())
            if i_iter % cfg.msg_iter == 0:
                ed = time.time()
                t_intv = ed - st
                msg = 'Fold: [{}/{}] Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                    'lr: {:.6f}, Loss: {:.6f}' .format(str(args.NowFold),args.Folds,
                        epoch, end_epoch, i_iter, epoch_iters1, 
                        t_intv, optimizer.param_groups[0]['lr'], sum(loss_avg) / len(loss_avg))
                if args.local_rank <= 0:
                    logging.info(msg)
        if args.local_rank <= 0:
            writer.add_scalar('train_loss', sum(loss_avg) / len(loss_avg), global_steps)
            writer.add_scalar('train_loss1', sum(loss1_avg) / len(loss1_avg), global_steps)
            writer.add_scalar('lr',optimizer.param_groups[0]['lr'],global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
            logger.info('=> saving checkpoint to {}'.format(
                args.resPath + 'checkpoint_'+str(args.NowFold)+'.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'best_mIoU': best_mIoU,
                'omit_list': omit_list,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.resPath,'checkpoint_'+str(args.NowFold)+'.pth.tar'))
        scheduler.step()
        torch.distributed.barrier()
    last_epoch = max(last_epoch, args.Delay)
    if -1 in omit_list:
        evaluator = MscEval(cfg, args, train_indexes[args.NowFold])
        val_loss_avg, val_mIOU, IoU_array, _ = evaluator(net)
        omit_list = torch.flip(torch.argsort(IoU_array),dims=[0])[:args.BestNum]
    for epoch in range(last_epoch, end_epoch):
        net.train()
        sampler2.set_epoch(epoch)
        loss_avg = []
        loss1_avg = []
        loss2_avg = []
        loss3_avg = []
        loss4_avg = []
        cur_iters = (epoch-args.Delay)*epoch_iters2
        if args.local_rank <= 0:
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
        st = time.time()
        for i_iter, batch in enumerate(dl2,0):
            images, labels, indexes, _, _ = batch
            images = images.cuda()
            labels = labels.cuda()
            indexes = indexes.cuda()
            labels = torch.squeeze(labels, 1)
            indexes = torch.squeeze(indexes, 1)
            optimizer.zero_grad()
            feature_out, logits = net(images)

            loss1 = criteria(logits, labels)
            loss2 = criteria1(feature_out, labels, indexes, omit_list)*args.lamda1
            loss3 = criteria2(feature_out, labels, indexes)*args.lamda2
            loss4 = criteria3(feature_out, labels, indexes, omit_list)*args.lamda3
            loss = loss1 + loss2 + loss3 + loss4
            adjust_learning_rate(optimizer,
                    args.lrStart2,
                    (end_epoch-args.Delay)*epoch_iters2,
                    i_iter+cur_iters,
                    )
            loss.backward()
            optimizer.step()

            loss_avg.append(loss.item())
            loss1_avg.append(loss1.item())
            loss2_avg.append(loss2.item())
            loss3_avg.append(loss3.item())
            loss4_avg.append(loss4.item())
            if i_iter % cfg.msg_iter == 0:
                ed = time.time()
                t_intv = ed - st
                msg = msg = 'Fold: [{}/{}] Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                    'lr: {:.6f}, Loss: {:.6f}' .format(str(args.NowFold),args.Folds,
                        epoch, end_epoch, i_iter, epoch_iters2, 
                        t_intv, optimizer.param_groups[0]['lr'], sum(loss_avg) / len(loss_avg))
                if args.local_rank <= 0:
                    logging.info(msg)

        if args.local_rank <= 0:
            writer.add_scalar('train_loss', sum(loss_avg) / len(loss_avg), global_steps)
            writer.add_scalar('train_loss1', sum(loss1_avg) / len(loss1_avg), global_steps)
            writer.add_scalar('train_loss2', sum(loss2_avg) / len(loss2_avg), global_steps)
            writer.add_scalar('train_loss3', sum(loss3_avg) / len(loss3_avg), global_steps)
            writer.add_scalar('train_loss4', sum(loss4_avg) / len(loss4_avg), global_steps)
            writer.add_scalar('lr',optimizer.param_groups[0]['lr'],global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

        evaluator = MscEval(cfg, args, train_indexes[args.NowFold])
        val_loss_avg, val_mIOU, IoU_array, _ = evaluator(net)
        omit_list = torch.flip(torch.argsort(IoU_array),dims=[0])[:args.BestNum]
        if args.local_rank <= 0:
            writer.add_scalar('valid_loss', val_loss_avg, global_steps)
            writer.add_scalar('valid_mIoU', val_mIOU, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

            if val_mIOU > best_mIoU:
                best_mIoU = val_mIOU
                torch.save(net.state_dict(),
                        os.path.join(args.resPath, 'best_'+str(args.NowFold)+'.pth'))
            logger.info('=> saving checkpoint to {}'.format(
                args.resPath + 'checkpoint_'+str(args.NowFold)+'.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'best_mIoU': best_mIoU,
                'omit_list': omit_list,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.resPath,'checkpoint_'+str(args.NowFold)+'.pth.tar'))
            msg = 'Loss: {:.3f}, MeanIoU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                        val_loss_avg, val_mIOU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)
        torch.distributed.barrier()
    if args.local_rank <= 0:
        torch.save(net.state_dict(),
                os.path.join(args.resPath, 'final_state_'+str(args.NowFold)+'.pth'))
        writer_dict['writer'].close()

if __name__ == "__main__":
    args = parse_args()
    if args.dataset == 'gid':
        cfg = config_factory['config_gid']
    else:
        cfg = config_factory['config_deepglobe']
    train(args=args, cfg=cfg)