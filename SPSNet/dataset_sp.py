import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
from PIL import Image
import numpy as np
import json
from transform import *

class LULCDataset(Dataset):
    def __init__(self, cfg, bashArgs, dataList, mode, indexes=None, *args, **kwargs):
        super(LULCDataset, self).__init__(*args, **kwargs)
        self.cfg = cfg
        self.bashArgs = bashArgs
        self.dataList = dataList
        self.mode = mode
        if self.bashArgs.dataset == 'gid':
            with open('./gid_info.json', 'r') as fr:
                labels_info = json.load(fr)
        else:
            with open('./deepglobe_info.json', 'r') as fr:
                labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        self.img_list = [line.strip().split() for line in open(os.path.join(self.bashArgs.Root,dataList))]
        
        self.files = []
        count = -1
        if self.mode == 'test':
            if self.bashArgs.dataset == 'gid':
                self.mean=torch.from_numpy(np.array([0.496,0.370,0.390,0.362])).float()
                self.std=torch.from_numpy(np.array([0.257,0.242,0.233,0.233])).float()
            else:
                self.mean=torch.from_numpy(np.array([0.24378921,0.34647351,0.38518094])).float()
                self.std=torch.from_numpy(np.array([0.10298847,0.12556953,0.16742616])).float()
            for item in self.img_list:
                count+=1
                if(indexes is not None):
                    if(count not in indexes):
                        continue
                image_path = item[0]
                name = os.path.splitext(os.path.basename(image_path))[0]
                self.files.append({
                    "img": image_path,
                    "name": name,
                })
                
        else:
            if self.bashArgs.dataset == 'gid':
                self.mean=torch.from_numpy(np.array([0.496,0.370,0.390,0.362])).float()
                self.std=torch.from_numpy(np.array([0.257,0.242,0.233,0.233])).float()
            else:
                self.mean=torch.from_numpy(np.array([0.28036855,0.37817039,0.4080014])).float()
                self.std=torch.from_numpy(np.array([0.10472486,0.11470858,0.14728342])).float()
            for item in self.img_list:
                count+=1
                if(indexes is not None):
                    if(count not in indexes):
                        continue
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(image_path))[0]
                self.files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        self.len = len(self.files)
        if self.bashArgs.dataset == 'gid':
            if self.bashArgs.ClassWeight:
                if self.bashArgs.Background:
                    self.class_weights = torch.FloatTensor([1.30675394,0.5360408,1.17171716,0.70942184,1.,1.]).cuda()
                else:
                    self.class_weights = torch.FloatTensor([1.30675394,0.5360408,1.17171716,0.70942184,1.]).cuda()
            else:
                self.class_weights = torch.ones(self.cfg.n_classes).cuda()
        else:
            if self.bashArgs.ClassWeight:
                if self.bashArgs.Background:
                    self.class_weights = torch.FloatTensor([1.14424052, 0.41540251, 1.45577277, 0.45897792, 1.98172387, 0.88805383, 1])
                else:
                    self.class_weights = torch.FloatTensor([1.14424052, 0.41540251, 1.45577277, 0.45897792, 1.98172387, 0.88805383])
            else:
                self.class_weights = torch.ones(self.cfg.n_classes).cuda()

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            ])
        transList = []
        if(self.bashArgs.HFTrans):
            transList.append(HorizontalFlip())
        if(self.bashArgs.VFTrans):
            transList.append(VerticalFlip())
        if(self.bashArgs.RRTrans):
            transList.append(RandomRotate((0,90)))
        if(self.bashArgs.RSTrans):
            transList.append(RandomScale(cfg.scales))
        if(self.bashArgs.RCTrans):
            transList.append(RandomCrop(cfg.crop_size))
        self.trans = Compose(transList)

    def __getitem__(self, idx):
        item = self.files[idx]
        fn = item['name']
        img = Image.open(os.path.join(self.bashArgs.Root,'img',item['img']))
        size = img.size
        if self.mode == 'test':
            image = self.input_transform(img)
            return image, np.array(size), fn
        
        
        if self.mode == 'train_sp':
            label = Image.open(os.path.join(self.bashArgs.Root,'sp',item['label']))
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans(im_lb)
            img, label = im_lb['im'], im_lb['lb']
            image = self.input_transform(img)
            label = np.array(label).astype(np.int64)
            sp_idx = label[:,:,0]+label[:,:,1]*256
            sp_idx = np.array(sp_idx)[np.newaxis, :]
            label = label[:,:,2]
            label = np.array(label)[np.newaxis, :]
            if self.bashArgs.Background:
                label = self.convert_labels(label)
            return image, label, sp_idx, np.array(size), fn
        else:
            label = Image.open(os.path.join(self.bashArgs.Root,'lb',item['label']))
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans(im_lb)
            img, label = im_lb['im'], im_lb['lb']
            image = self.input_transform(img)
            label = np.array(label).astype(np.int64)
            label = np.array(label)[np.newaxis, :]
            if self.bashArgs.Background:
                label = self.convert_labels(label)
            return image, label, np.array(size), fn

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        temp = label.copy()
        for k, v in self.lb_map.items():
            label[temp == k] = v
        return label
    
    def input_transform(self, image):
        image = self.to_tensor(image)
        image = image.permute((1,2,0))
        image = (image - self.mean)
        image = (image / self.std)
        image = image.permute((2,0,1))
        return image
