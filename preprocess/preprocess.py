import argparse
from PIL import Image
import numpy as np
import cv2, os, math
import multiprocessing
from functools import partial

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-d','--dataset',required=True,type=str,choices=['GID','DeepGlobe'])
    parse.add_argument('-i','--imgpath',required=True,type=str)
    parse.add_argument('-l','--labpath',required=True,type=str)
    parse.add_argument('-s','--savepath',required=True,type=str)
    return parse.parse_args()

def LabelTransGid(label):
    label_map = [255,0,1,2,3,4]
    label_trans = label.copy()
    for i in range(len(label_map)):
        label_trans[label == i] = label_map[i]
    return label_trans

def LabelTransDeepglobe(label):
    label_trans = np.ones((label.shape[0],label.shape[1])).astype('uint8')*255
    r,g,b = label[:,:,0],label[:,:,1],label[:,:,2]
    b_0 = b<128
    b_255 = b>=128
    g_0 = g<128
    g_255 = g>=128
    r_0 = r<128
    r_255 = r>=128
    label_trans[(r_0*g_255*b_255)]=0
    label_trans[(r_255*g_255*b_0)]=1
    label_trans[(r_255*g_0*b_255)]=2
    label_trans[(r_0*g_255*b_0)]=3
    label_trans[(r_0*g_0*b_255)]=4
    label_trans[(r_255*g_255*b_255)]=5
    return label_trans

def Resample(label, class_count):
    lab_resample = label.copy()
    interval = math.floor(255 / class_count)
    for i in range(1, class_count):
        lab_resample[label == i] = interval * i
    lab_resample[label==255] = interval * class_count
    return lab_resample

def LscSp(img_lab, lab, sp_size, dataset):
    if dataset=='gid':
        class_count = 5
    else:
        class_count = 6
    lsc = cv2.ximgproc.createSuperpixelLSC(img_lab,region_size=sp_size)
    lsc.iterate(10)
    lsc.enforceLabelConnectivity(sp_size)
    label_lsc = lsc.getLabels()+1
    number_lsc = lsc.getNumberOfSuperpixels()
    lsc_split_tmp = np.zeros((img_lab.shape[0],img_lab.shape[1],class_count)) # sp segment by class
    sp_out = np.zeros((img_lab.shape[0],img_lab.shape[1],3)) # class: channel B, id num: channel R+G*256 :
    sp_out[lab == 255]=(255,0,0)
    if dataset == 'gid':
        for i in range(class_count):
            if i == 1 or i == 2 or i == 4:
                continue
            lab_single = np.zeros((lab.shape[0],lab.shape[1]),dtype='uint8')
            lab_single[lab==i] = 1
            _, patches, _, _ = cv2.connectedComponentsWithStats(lab_single, connectivity=4)
            sp_out[:,:,0][patches>0] = i
            sp_out[:,:,1][patches>0] = patches[patches>0]//256
            sp_out[:,:,2][patches>0] = patches[patches>0]%256
        
    for i in range(class_count):
        lsc_split_tmp[:,:,i][lab==i]=1
        lsc_split_tmp[:,:,i] *= label_lsc
    for i in range(class_count):
        id_start = 1
        lsc_split = np.zeros((img_lab.shape[0],img_lab.shape[1]))
        if not (dataset=='gid' and (i == 0 or i == 3)):
            for j in range(1,number_lsc+1):
                if j in lsc_split_tmp[:,:,i]:
                    lsc_split[lsc_split_tmp[:,:,i]==j] = id_start
                    id_start += 1
            sp_out[:,:,0][lsc_split>0] = i
            sp_out[:,:,1][lsc_split>0] = lsc_split[lsc_split>0]//256
            sp_out[:,:,2][lsc_split>0] = lsc_split[lsc_split>0]%256
    return sp_out

def PreProcess(img, lab, patchsize, overlap, savepath, id, dataset, sp_size=50):
    row = img.shape[0]
    col = img.shape[1]
    i_total = math.floor((row - overlap) / (patchsize - overlap))
    j_total = math.floor((col - overlap) / (patchsize - overlap))
    if dataset=='gid':
        profix1='.tif'
        lab_resample = Resample(lab, 5)
    else:
        profix1='.png'
        lab_resample = Resample(lab, 6)
    profix2='_class.png'

    img_lab = np.concatenate((img, lab_resample[:,:,np.newaxis]),2)

    for i in range(i_total):
        for j in range(j_total):
            img_split = img[i * patchsize - i * overlap:(i + 1) * patchsize - i * overlap, j * patchsize - j * overlap:(j + 1) * patchsize - j * overlap]
            lab_split = lab[i * patchsize - i * overlap:(i + 1) * patchsize - i * overlap, j * patchsize - j * overlap:(j + 1) * patchsize - j * overlap]
            if (lab_split == 255).all():
                continue
            img_savename = os.path.join(savepath, 'img', id+'_'+str(patchsize)+ "_" + str(i * patchsize - i * overlap) + "_" + str(j * patchsize - j * overlap) +profix1)
            img_save = Image.fromarray(img_split)
            img_save.save(img_savename)
            del img_save

            lab_savename = os.path.join(savepath, 'lb', id+'_'+str(patchsize)+ "_" + str(i * patchsize - i * overlap) + "_" + str(j * patchsize - j * overlap) +profix2)
            lab_save = Image.fromarray(lab_split)
            lab_save.save(lab_savename)
            del lab_save

            img_lab_split = img_lab[i * patchsize - i * overlap:(i + 1) * patchsize - i * overlap, j * patchsize - j * overlap:(j + 1) * patchsize - j * overlap]
            sp = LscSp(img_lab_split, lab_split, 50, dataset)
            sp_savename = os.path.join(savepath, 'sp', id+'_'+str(patchsize)+ "_" + str(i * patchsize - i * overlap) + "_" + str(j * patchsize - j * overlap) +profix2)
            cv2.imwrite(sp_savename,sp)
            del sp

def ProcessSingleImage(args, img_profix, lab_profix, id):
    img = np.array(Image.open(os.path.join(args.imgpath,id+img_profix)))
    lab = np.array(Image.open(os.path.join(args.labpath,id+lab_profix)))
    if args.dataset=='GID':
        lab = LabelTransGid(lab)
        PreProcess(img, lab, 1024, 0, args.savepath, id, 'gid')
    else:
        lab = LabelTransDeepglobe(lab)
        PreProcess(img, lab, 768, 208, args.savepath, id, 'deepglobe')

if __name__=='__main__':
    args = parse_args()
    if not os.path.exists(os.path.join(args.savepath,'img')):
        os.makedirs(os.path.join(args.savepath,'img'))
    if not os.path.exists(os.path.join(args.savepath,'lb')):
        os.makedirs(os.path.join(args.savepath,'lb'))
    if not os.path.exists(os.path.join(args.savepath,'sp')):
        os.makedirs(os.path.join(args.savepath,'sp'))
    imgnames = os.listdir(args.imgpath)
    if args.dataset=='GID':
        img_profix = '.tif'
        lab_profix = '_label.png'
        img_ids = [name[:-4] for name in imgnames]
    else:
        img_profix = '_sat.jpg'
        lab_profix = '_mask.png'
        img_ids = [name[:-8] for name in imgnames if img_profix in name]
    
    func = partial(ProcessSingleImage, args, img_profix, lab_profix)

    pool = multiprocessing.Pool(processes=8)
    pool.map(func, img_ids)
    pool.close()
    pool.join()
