import argparse
from PIL import Image
import numpy as np
import cv2, os, math, shutil
import multiprocessing
from functools import partial

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('-i','--imgpath',required=True,type=str)
    parse.add_argument('-l','--labpath',required=True,type=str)
    parse.add_argument('-p','--listpath',required=True,type=str)
    parse.add_argument('-s','--savepath',required=True,type=str)
    return parse.parse_args()

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

def ProcessSingleImage(args, line):
    if len(line) > 0:
        img_profix = '_sat.jpg'
        lab_profix = '_mask.png'
        id = line.split()[0][:-4]
        shutil.copyfile(os.path.join(args.imgpath, id+img_profix),os.path.join(args.savepath,'img', id+img_profix[-4:]))
        lab = np.array(Image.open(os.path.join(args.labpath,id+lab_profix)))
        lab = LabelTransDeepglobe(lab)
        lab_savename = os.path.join(args.savepath,'lb', id+'_class.png')
        lab_save = Image.fromarray(lab)
        lab_save.save(lab_savename)

if __name__=='__main__':
    args = parse_args()
    img_savepath = os.path.join(args.savepath,'img')
    lab_savepath = os.path.join(args.savepath,'lb')
    if not os.path.exists(img_savepath):
        os.makedirs(img_savepath)
    if not os.path.exists(lab_savepath):
        os.makedirs(lab_savepath)

    lst = open(args.listpath,'r')
    content = lst.readlines()

    func = partial(ProcessSingleImage, args)
    pool = multiprocessing.Pool(processes=4)
    pool.map(func, content)
    pool.close()
    pool.join()
    
    shutil.copyfile(args.listpath, os.path.join(args.savepath,os.path.basename(args.listpath)))