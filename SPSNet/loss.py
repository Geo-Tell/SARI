import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.optim import optimizer


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, class_weight=None, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_lb)

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_pixs = N * H * W
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        with torch.no_grad():
            scores = F.softmax(logits, dim=1)
            labels_cpu = labels
            invalid_mask = labels_cpu==self.ignore_lb
            labels_cpu[invalid_mask] = 0
            picks = scores[torch.arange(n_pixs), labels_cpu]
            picks[invalid_mask] = 1
            sorteds, _ = torch.sort(picks)
            thresh = self.thresh if sorteds[self.n_min]<self.thresh else sorteds[self.n_min]
            labels[picks>thresh] = self.ignore_lb
        labels = labels.clone()
        loss = self.criteria(logits, labels)
        return loss

class Lvar(nn.Module):
    def __init__(self, n_classes=5, ignore_lb=255):
        super(Lvar, self).__init__()
        self.n_classes = n_classes
        self.ignore_lb = ignore_lb
    def forward(self, feature_out, labels, indexes, omit_list=[]):
        loss = torch.zeros(1).cuda()
        feature_out = feature_out.permute((0,2,3,1)) # [B, H, W, D]
        B, _, _, D = feature_out.shape

        for n in range(B):
            label_single = labels[n,:,:] # [H,W]
            feat_single = feature_out[n,:,:,:] # [H,W,D]
            index_single = indexes[n,:,:] # [H,W]
            max = index_single.max()

            feat_single = feat_single.reshape(-1, D)  # [HW, D]
            with torch.no_grad():
                label_single_temp = label_single.clone().reshape(-1)
                index_single_temp = index_single.clone().reshape(-1)
                index_single_temp[label_single_temp == self.ignore_lb] = 0
                label_single_temp[label_single_temp == self.ignore_lb] = 0
                index_single_new = max * label_single_temp + index_single_temp
                sorted, indicies = torch.sort(index_single_new)                         # [00011122]
                sorted_temp = torch.cat((sorted.clone()[1:], sorted[-1].unsqueeze(0)))  # [00111222]
                # (start index - 1) of each superpixel                                       ^  ^
                index_range = torch.nonzero(sorted-sorted_temp)                         #    i  i
            
            if index_range.shape[0] == 0:
                if sorted[0] == 0:  # no superpixel
                    continue
                else:               # only 1 superpixel
                    # class not in omit list
                    if torch.ceil(sorted[0].float() / max - 1).long() not in omit_list:
                        # compute variance
                        loss += ((feat_single[indicies[:]]
                            - torch.mean(feat_single[indicies[:]],0).detach())**2).mean(0).mean()*5
                    else:
                        continue
            else:   # more than 1 superpixel
                sp_mean = torch.zeros(index_range.shape[0] if sorted[0] == 0 else index_range.shape[0]+1, D).cuda()
                sp_var = torch.zeros(index_range.shape[0] if sorted[0] == 0 else index_range.shape[0]+1).cuda()
                # variance in the first superpixel if sorted[0] != 0
                if sorted[0] != 0 and index_range[0] > 0 and torch.ceil(sorted[0].float() / max - 1).long() not in omit_list:
                    sp_mean[-2] = torch.mean(feat_single[indicies[:index_range[0]+1]],0)  # [1, D]
                    sp_var[-2] = ((feat_single[indicies[:index_range[0]+1]] - sp_mean[-2].detach())**2).mean(0).mean()*5
                # variance in all superpixels
                for i in range(index_range.shape[0]-1):
                    if (index_range[i+1] - index_range[i] > 1) and (torch.ceil(sorted[index_range[i]+1].float() / max - 1).long() not in omit_list):
                        sp_mean[i] = torch.mean(feat_single[indicies[index_range[i]+1:index_range[i+1]+1]],0)
                        sp_var[i] = ((feat_single[indicies[index_range[i]+1:index_range[i+1]+1]] - sp_mean[i].detach())**2).mean(0).mean()*5
                # variance in the last superpixel
                if sorted[-2]-sorted_temp[-2] == 0 and torch.ceil(sorted[-1].float() / max - 1).long() not in omit_list:
                    sp_mean[-1] = torch.mean(feat_single[indicies[index_range[-1]+1:]],0)
                    sp_var[-1] = ((feat_single[indicies[index_range[-1]+1:]] - sp_mean[-1].detach())**2).mean(0).mean()*5
                loss += torch.sum(sp_var)
        return torch.sqrt(loss)

class Linter(nn.Module):
    def __init__(self, n_classes=5, ignore_lb=255):
        super(Linter, self).__init__()
        self.n_classes = n_classes
        self.ignore_lb = ignore_lb
    def forward(self, feature_out, labels, indexes):
        loss = torch.zeros(1).cuda()
        feature_out = feature_out.permute((0,2,3,1)) # [B, H, W, D]
        B, _, _, D = feature_out.shape
        losses = []
        for n in range(B):
            label_single = labels[n,:,:] # [H,W]
            feat_single = feature_out[n,:,:,:] # [H,W,D]
            index_single = indexes[n,:,:] # [H,W]
            max = index_single.max()
            feat_single = feat_single.reshape(-1, D)
            with torch.no_grad():
                label_single_temp = label_single.clone().reshape(-1)
                index_single_temp = index_single.clone().reshape(-1)
                index_single_temp[label_single_temp == self.ignore_lb] = 0
                label_single_temp[label_single_temp == self.ignore_lb] = 0
                index_single_new = max * label_single_temp + index_single_temp
                sorted, indicies = torch.sort(index_single_new)                         # [00011122]
                sorted_temp = torch.cat((sorted.clone()[1:], sorted[-1].unsqueeze(0)))  # [00111222]
                # (start index - 1) of each superpixel                                       ^  ^
                index_range = torch.nonzero(sorted-sorted_temp)                         #    i  i
            feat_class = torch.ones(0,feat_single.shape[-1]).cuda() # [0, D]
            feat_classes = []
            if index_range.shape[0] == 0:   # 0 or 1 superpixel
                continue
            else:   # more than 1 superpixel
                last_class = self.ignore_lb
                if sorted[0] != 0 and index_range[0] > 0:
                    # record average feature of the first superpixel if sorted[0] != 0
                    feat_class = torch.cat((feat_class,torch.mean(feat_single[indicies[:index_range[0]+1]],0).unsqueeze(0)),0)
                    last_class = torch.ceil(sorted[0].float() / max - 1)
                # record average feature of all superpixels
                for i in range(index_range.shape[0]-1):
                    if index_range[i+1] - index_range[i] > 1:
                        current_class = torch.ceil(sorted[index_range[i]+1].float() / max - 1)
                        if current_class != last_class and feat_class.shape[0]>0:
                            feat_classes.append(feat_class)
                            feat_class = torch.ones(0,feat_single.shape[-1]).cuda() # [0, 64]
                        feat_class = torch.cat((feat_class,torch.mean(feat_single[indicies[index_range[i]+1:index_range[i+1]+1]],0).unsqueeze(0)),0)
                        last_class = current_class
                # record average feature of the last superpixel
                if sorted[-2]-sorted_temp[-2] == 0:
                    current_class = torch.ceil(sorted[index_range[-1]].float() / max - 1)
                    if current_class != last_class and feat_class.shape[0]>0:
                        feat_classes.append(feat_class)
                        feat_class = torch.ones(0,feat_single.shape[-1]).cuda() # [0, 64]
                    feat_class = torch.cat((feat_class,torch.mean(feat_single[indicies[index_range[-1]+1:]],0).unsqueeze(0)),0)
                if feat_class.shape[0]>0:
                    feat_classes.append(feat_class)

                for i in range(len(feat_classes)-1):
                    for j in range(i+1,len(feat_classes)):
                        if feat_classes[i] is not None and feat_classes[j] is not None:
                            x = feat_classes[i].reshape((feat_classes[i].shape[0],1,D))  # [N1, 1, D]
                            y = feat_classes[j].reshape((1,feat_classes[j].shape[0],D))  # [1, N2, D]
                            # smoothL1 distance:
                            ret = torch.mean(torch.abs(x-y))
                            ret = torch.where(ret < 1, 0.5 * ret ** 2, ret - 0.5)
                            losses.append(ret)

        if len(losses)>0:
            loss -= torch.mean(torch.stack(losses))
        if loss == 0:
            loss = -torch.ones((1)).cuda().detach() * B
        return -torch.log(-loss/B)

class Lintra(nn.Module):
    def __init__(self, n_classes=5, ignore_lb=255):
        super(Lintra, self).__init__()
        self.n_classes = n_classes
        self.ignore_lb = ignore_lb
    def forward(self, feature_out, labels, indexes, omit_list=[]):
        loss = torch.zeros(1).cuda()
        feature_out = feature_out.permute((0,2,3,1)) # [B, H, W, D]
        B, _, _, D = feature_out.shape
        losses = []
        for n in range(B):
            label_single = labels[n,:,:] # [H,W]
            feat_single = feature_out[n,:,:,:] # [H,W,D]
            index_single = indexes[n,:,:] # [H,W]
            max = index_single.max()
            feat_single = feat_single.reshape(-1, D)
            with torch.no_grad():
                label_single_temp = label_single.clone().reshape(-1)
                index_single_temp = index_single.clone().reshape(-1)
                index_single_temp[label_single_temp == self.ignore_lb] = 0
                label_single_temp[label_single_temp == self.ignore_lb] = 0
                index_single_new = max * label_single_temp + index_single_temp
                sorted, indicies = torch.sort(index_single_new)                         # [00011122]
                sorted_temp = torch.cat((sorted.clone()[1:], sorted[-1].unsqueeze(0)))  # [00111222]
                # (start index - 1) of each superpixel                                       ^  ^
                index_range = torch.nonzero(sorted-sorted_temp)                         #    i  i
            feat_class = torch.ones(0,feat_single.shape[-1]).cuda() # [0, D]
            feat_classes = []
            if index_range.shape[0] == 0:   # 0 or 1 superpixel
                continue
            else:   # more than 1 superpixel
                last_class = self.ignore_lb
                current_class = torch.ceil(sorted[0].float() / max - 1).long()
                if sorted[0] != 0 and index_range[0] > 0 and current_class not in omit_list:
                    feat_class = torch.cat((feat_class,torch.mean(feat_single[indicies[:index_range[0]+1]],0).unsqueeze(0)),0)
                    last_class = current_class
                # record average feature of all superpixels
                for i in range(index_range.shape[0]-1):
                    if index_range[i+1] - index_range[i] > 1:
                        current_class = torch.ceil(sorted[index_range[i]+1].float() / max - 1).long()
                        if current_class != last_class and feat_class.shape[0]>0:
                            feat_classes.append(feat_class)
                            feat_class = torch.ones(0,feat_single.shape[-1]).cuda() # [0, 64]
                        if current_class not in omit_list:
                            feat_class = torch.cat((feat_class,torch.mean(feat_single[indicies[index_range[i]+1:index_range[i+1]+1]],0).unsqueeze(0)),0)
                        last_class = current_class
                # record average feature of the last superpixel
                if sorted[-2]-sorted_temp[-2] == 0:
                    current_class = torch.ceil(sorted[index_range[-1]].float() / max - 1).long()
                    if current_class != last_class and feat_class.shape[0]>0:
                        feat_classes.append(feat_class)
                        feat_class = torch.ones(0,feat_single.shape[-1]).cuda() # [0, 64]
                    if current_class not in omit_list:
                        feat_class = torch.cat((feat_class,torch.mean(feat_single[indicies[index_range[-1]+1:]],0).unsqueeze(0)),0)
                if feat_class.shape[0]>0:
                    feat_classes.append(feat_class)

            for i in range(len(feat_classes)):
                if(feat_classes[i] is not None):
                    x = feat_classes[i].reshape((feat_classes[i].shape[0],1,D))  # [N1, 1, D]
                    y = feat_classes[i].reshape((1,feat_classes[i].shape[0],D))  # [1, N2, D]
                    # smoothL1 distance
                    ret = torch.mean(torch.abs(x-y))
                    ret = torch.where(ret < 1, 0.5 * ret ** 2, ret - 0.5)
                    losses.append(ret)
        if len(losses)>0:
            loss += torch.mean(torch.stack(losses))
        else:
            loss += torch.zeros((1)).cuda().detach() * B

        return loss/B