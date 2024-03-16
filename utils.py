import os
import clip
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys

import cv2
import glob
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def load_text_feature(cfg):
    save_path = cfg['cache_dir'] + "/text_weights_template.pt"
    clip_weights_template = torch.load(save_path)
    save_path = cfg['cache_dir'] + "/text_weights_cupl.pt"
    clip_weights_cupl = torch.load(save_path)
    save_path = cfg['cache_dir'] + "/text_weights_neutral_template.pt"
    clip_weights_neutral = torch.load(save_path)
    return clip_weights_template, clip_weights_cupl, clip_weights_neutral


def load_few_shot_feature(cfg):
    cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
    cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
    return cache_keys, cache_values


def loda_val_test_feature(cfg, split):
    features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
    labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    return features, labels

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def accuracy(shot_logits, cache_values, topk=(1,)):
    target = cache_values.topk(max(topk), 1, True, True)[1].squeeze()
    pred = shot_logits.topk(max(topk), 1, True, True)[1].squeeze()
    idx = (target != pred)
    return idx

def kl_loss(logit1, logit2):
    p = F.log_softmax(logit1, dim=1)
    q = F.softmax(logit2, dim=1) + 1e-8
    # print(p)
    # print(q)
    kl_div = nn.KLDivLoss(reduction='none')
    kl = kl_div(p, q).sum(dim=1)
    # print(kl.shape)
    # print(kl)
    return kl.mean()


def generate_pseudo_negative_cache(cfg, cache_keys):
    print('Generating pseudo negative cache...')
    num_shot = cfg['shots']
    num_class = cache_keys.shape[0] // cfg['shots']
    feat_dim = cache_keys.shape[-1]

    
    # Reshaping the cache keys
    cache_keys = cache_keys.reshape(num_class, num_shot, feat_dim)
    
    # Initializing negative cache keys
    negative_cache_keys = torch.zeros((num_class, num_shot, feat_dim), device='cuda')

    filtered = 1
    num_negative = num_class - filtered

    # Precompute mean cache keys for each class
    mean_cache_keys = cache_keys.mean(dim=1)
    mean_cache_keys = F.normalize(mean_cache_keys, dim=1)

    # Compute all cosine similarities in a vectorized manner
    similarity_matrix = mean_cache_keys @ mean_cache_keys.t()

    # Get indices of the classes with lowest similarity
    _, negative_indices = torch.topk(similarity_matrix, k=num_negative, largest=False, dim=1)

    # Calculate negative cache keys
    for i in range(num_class):
        selected_cache_keys = cache_keys[negative_indices[i, :], :, :]
        negative_cache_keys[i, :, :] = torch.mean(selected_cache_keys, dim=0)

    # Reshape and normalize
    negative_cache_keys = negative_cache_keys.reshape(-1, feat_dim)
    negative_cache_keys = F.normalize(negative_cache_keys, dim=1)
    
    return negative_cache_keys

def generate_soft_label(cfg, cache_keys, cache_values, temperature=0.1):
    num_shot = cfg['shots']
    num_class = cache_keys.shape[0] // cfg['shots']
    feat_dim = cache_keys.shape[-1]
    
    if num_shot == 1:
        return cache_values.half()
    else:
        # Reshaping the cache keys and values
        cache_keys = cache_keys.reshape(num_class, num_shot, feat_dim)
        cache_values = cache_values.reshape(num_class, num_shot, num_class)
        
        soft_cache_values = torch.zeros((num_class, num_shot, num_class), device='cuda')
        for i in range(num_class):
            keys = cache_keys[i, :, :]
            values = cache_values[i, :, :]
            cos_sim = keys @ keys.t()
            sum_sim = cos_sim.sum(dim=1) - 1
            avg_sim = sum_sim / (num_shot - 1)
            confidence = F.softmax(avg_sim / temperature, dim=0)
            soft_cache_values[i, :, :] = values * confidence.unsqueeze(1) * num_shot
        
        soft_cache_values = soft_cache_values.reshape(-1, num_class)

        return soft_cache_values.half()

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()
    
class PositiveAdapter(nn.Module):
    def __init__(self, cfg, clip_weights_template, clip_weights_cupl, clip_weights_neutral, clip_model, cache_keys,negative_cache_keys, cache_values):
        super(PositiveAdapter, self).__init__()
        self.shots = cfg['shots']
        self.feat_dim, self.cate_num = clip_weights_template.shape
        self.feat_num = cfg['training_feat_num']
        self.indices = torch.arange(self.feat_dim).cuda()
        
        self.res_template = nn.Parameter(torch.zeros([self.cate_num, cfg['training_feat_num']]).half().cuda(), requires_grad=True)
        self.res_cupl = nn.Parameter(torch.zeros([self.cate_num, cfg['training_feat_num']]).half().cuda(), requires_grad=True)
        self.res_keys = nn.Parameter(torch.zeros([self.cate_num, cfg['training_feat_num']]).half().cuda(), requires_grad=True)

        
    def forward(self, cache_keys, negative_cache_keys, clip_weights_template, clip_weights_cupl, clip_weights_neutral, cache_values):
        new_cache_keys = cache_keys.clone()
        new_cache_keys = new_cache_keys.reshape(-1, self.feat_dim)
        new_cache_keys = new_cache_keys + self.res_keys.unsqueeze(1).repeat(1, self.shots, 1).reshape(-1, self.feat_num)
        new_cache_values = cache_values
        
        new_negative_cache_keys = negative_cache_keys.clone()
        new_negative_cache_keys = new_negative_cache_keys.reshape(-1, self.feat_dim)

        res_text_template = self.res_template.t()
        res_text_cupl = self.res_cupl.t()
        new_clip_weights_template = clip_weights_template.clone()
        new_clip_weights_template = clip_weights_template + res_text_template
        new_clip_weights_cupl = clip_weights_cupl.clone()
        new_clip_weights_cupl = clip_weights_cupl + res_text_cupl
        new_clip_weights_neutral = clip_weights_neutral.clone()
        
        # Normalize
        new_clip_weights_template = F.normalize(new_clip_weights_template, dim=0)
        new_clip_weights_cupl = F.normalize(new_clip_weights_cupl, dim=0)
        new_clip_weights_neutral = F.normalize(new_clip_weights_neutral, dim=0)
        new_cache_keys = F.normalize(new_cache_keys, dim=1)
        new_negative_cache_keys = F.normalize(new_negative_cache_keys, dim=1)
        
        return new_cache_keys.half(), new_negative_cache_keys.half(), new_clip_weights_template.half(), new_clip_weights_cupl.half(), new_clip_weights_neutral.half(), new_cache_values.half()
    
class NegativeAdapter(nn.Module):
    def __init__(self, cfg, clip_weights_template, clip_weights_cupl, clip_weights_neutral, clip_model, cache_keys,negative_cache_keys, cache_values):
        super(NegativeAdapter, self).__init__()
        self.shots = cfg['shots']
        self.feat_dim, self.cate_num = clip_weights_template.shape
        self.feat_num = cfg['training_feat_num']
        self.indices = torch.arange(self.feat_dim).cuda()
        
        self.res_neutral = nn.Parameter(torch.zeros([self.cate_num, cfg['training_feat_num']]).half().cuda(), requires_grad=True)
        self.res_keys2 = nn.Parameter(torch.zeros([self.cate_num, cfg['training_feat_num']]).half().cuda(), requires_grad=True)

        
    def forward(self, cache_keys, negative_cache_keys, clip_weights_template, clip_weights_cupl, clip_weights_neutral, cache_values):
        new_cache_keys = cache_keys.clone()
        new_cache_keys = new_cache_keys.reshape(-1, self.feat_dim)
        new_cache_values = cache_values
        
        new_negative_cache_keys = negative_cache_keys.clone()
        new_negative_cache_keys = new_negative_cache_keys.reshape(-1, self.feat_dim)
        new_negative_cache_keys = new_negative_cache_keys + self.res_keys2.unsqueeze(1).repeat(1, self.shots, 1).reshape(-1, self.feat_num)

        res_text_neutral = self.res_neutral.t()
        new_clip_weights_template = clip_weights_template.clone()
        new_clip_weights_cupl = clip_weights_cupl.clone()
        new_clip_weights_neutral = clip_weights_neutral.clone()
        new_clip_weights_neutral = clip_weights_neutral + res_text_neutral
        # Normalize
        new_clip_weights_template = F.normalize(new_clip_weights_template, dim=0)
        new_clip_weights_cupl = F.normalize(new_clip_weights_cupl, dim=0)
        new_clip_weights_neutral = F.normalize(new_clip_weights_neutral, dim=0)
        new_cache_keys = F.normalize(new_cache_keys, dim=1)
        new_negative_cache_keys = F.normalize(new_negative_cache_keys, dim=1)
        
        return new_cache_keys.half(), new_negative_cache_keys.half(), new_clip_weights_template.half(), new_clip_weights_cupl.half(), new_clip_weights_neutral.half(), new_cache_values.half()