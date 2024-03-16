import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', dest='shot', type=int, default=1, help='shots number')
    parser.add_argument('--config', dest='config', help='settings of DualAdapter in yaml format')
    parser.add_argument('--test', dest='test', help='test dataset')
    args = parser.parse_args()
    return args


def DualAdapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights_template, clip_weights_cupl, clip_weights_neutral, clip_model, train_loader_F):
    
    feat_dim, cate_num = clip_weights_template.shape
    cache_values = cache_values.reshape(cate_num, -1, cate_num)
    cache_keys = cache_keys.t().reshape(cate_num, cfg['shots'], feat_dim).reshape(cate_num, -1, feat_dim)
    
    cfg['w'] = cfg['w_training']
    cache_keys, cache_values = cache_keys.reshape(-1, feat_dim), cache_values.reshape(-1, cate_num)
    print("**** cache_keys shape: {:}. ****\n".format(cache_keys.shape))
    print("**** cache_values shape: {:}. ****\n".format(cache_values.shape))
    negative_cache_keys = generate_pseudo_negative_cache(cfg, cache_keys)
    soft_cache_values = generate_soft_label(cfg, cache_keys, cache_values)
    positive_adapter = PositiveAdapter(cfg, clip_weights_template, clip_weights_cupl, clip_weights_neutral, clip_model, cache_keys, negative_cache_keys, cache_values).cuda()
    negative_adapter = NegativeAdapter(cfg, clip_weights_template, clip_weights_cupl, clip_weights_neutral, clip_model, cache_keys, negative_cache_keys, cache_values).cuda()
    
    optimizer = torch.optim.AdamW([
        {'params': positive_adapter.parameters(), 'lr': cfg['lr'], 'eps': cfg['eps'], 'weight_decay': 1e-1},
        {'params': negative_adapter.parameters(), 'lr': cfg['lr'] * 5, 'eps': cfg['eps'], 'weight_decay': 1e-1}
        ])  # negative adapter's learning rate is 5 times of positive adapter's learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    Loss = SmoothCrossEntropy()
    
    beta, alpha, beta2, alpha2, lam = cfg['init_beta'], cfg['init_alpha'], cfg['init_beta2'], cfg['init_alpha2'], cfg['init_lambda']
    best_acc, best_epoch = 0.0, 0
    feat_num = cfg['training_feat_num'] # feat_num

    new_clip_weights = 0.45 * clip_weights_template + 0.55 * clip_weights_cupl
    R_fW = 100. * (val_features @ new_clip_weights) 
    acc = cls_acc(R_fW, val_labels)
    print('(+) Zero-Shot CLIP\'s Accuracy: {:.2f}'.format(acc))
    
    for train_idx in range(cfg['train_epoch']):
        # Train
        positive_adapter.train()
        negative_adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        loss1_list = []
        loss2_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_neutral, R_FW = positive_adapter(cache_keys, negative_cache_keys, clip_weights_template, clip_weights_cupl, clip_weights_neutral, cache_values)
            new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_neutral, R_FW = negative_adapter(new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_neutral, R_FW)
            
            # Positive
            R_fF = image_features @ new_cache_keys.half().t()
            
            Aff = ((-1) * (beta - beta * (R_fF))).exp()
            cache_logits = Aff @ soft_cache_values
            new_clip_weights = 0.45 * new_clip_weights_template + 0.55 * new_clip_weights_cupl
            R_fW = 100. * (image_features @ new_clip_weights) 
            
            # Negative
            R_fF2 = (1 - image_features @ new_negative_cache_keys.half().t())
            Aff2 = ((-1) * (beta - beta * (R_fF2))).exp()
            cache_logits2 = Aff2 @ cache_values
            R_fW2 = 100. * (1 - image_features @ new_clip_weights_neutral) * 0.15 # to scale
            
            cos = nn.CosineSimilarity(dim=1, eps=1e-7)
            text_distance_template = 1 - cos(new_clip_weights, clip_weights_template)
            text_distance_cupl = 1 - cos(new_clip_weights, clip_weights_cupl)
            consistency_loss = (0.45 * text_distance_template + (1 - 0.45) * text_distance_cupl).mean()

            ape_logits = R_fW + cache_logits * alpha
            ape_logits2 = R_fW2 + cache_logits2 * alpha
            final_logits = lam * ape_logits + (1 - lam) * ape_logits2
            
            loss1 = Loss(final_logits, target)
            loss2 = adaptive_reranking_loss(image_features, new_clip_weights.t(), target)
            
            loss = loss1 + loss2 + 8 * consistency_loss# + loss3 #+ loss4

            acc = cls_acc(final_logits, target)
            correct_samples += acc / 100 * len(final_logits)
            all_samples += len(final_logits)
            loss_list.append(loss.item())
            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}, CE Loss: {:.4f}, ReRank Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list), sum(loss1_list)/len(loss1_list), sum(loss2_list)/len(loss2_list)))

        # Eval
        positive_adapter.eval()
        negative_adapter.eval()
        with torch.no_grad():
            R_fW_original = 100. * (val_features @ (0.45 * clip_weights_template + 0.55 * clip_weights_cupl))
            new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_neutral, R_FW = positive_adapter(cache_keys, negative_cache_keys, clip_weights_template, clip_weights_cupl, clip_weights_neutral, cache_values)
            new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_neutral, R_FW = negative_adapter(new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_neutral, R_FW)

            # Positive
            R_fF = val_features @ new_cache_keys.half().t()
            Aff = ((-1) * (beta - beta * (R_fF))).exp()
            cache_logits = Aff @ soft_cache_values
            new_clip_weights = 0.45 * new_clip_weights_template + 0.55 * new_clip_weights_cupl
            R_fW = 100. * (val_features @ new_clip_weights) 
            
            # Negative
            R_fF2 = (1 - val_features @ new_negative_cache_keys.half().t())
            Aff2 = ((-1) * (beta - beta * (R_fF2))).exp()
            cache_logits2 = Aff2 @ cache_values
            R_fW2 = 100. * (1 - val_features @ new_clip_weights_neutral) * 0.15
            
        
            ape_logits = R_fW + cache_logits * alpha
            ape_logits2 = R_fW2 + cache_logits2 * alpha
            final_logits = lam * ape_logits + (1 - lam) * ape_logits2
            
        acc = cls_acc(final_logits, val_labels)
        
        print("**** DualAdapter's test accuracy: {:.2f}. ****".format(acc))

        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(positive_adapter, cfg['cache_dir'] + "/positive_" + str(cfg['shots']) + "shots.pt")
            torch.save(negative_adapter, cfg['cache_dir'] + "/negative_" + str(cfg['shots']) + "shots.pt")
    positive_adapter = torch.load(cfg['cache_dir'] + "/positive_" + str(cfg['shots']) + "shots.pt")
    negative_adapter = torch.load(cfg['cache_dir'] + "/negative_" + str(cfg['shots']) + "shots.pt")

    print(f"**** DualAdapter's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")
    
    print("\n-------- Searching hyperparameters on the val set. --------")
    # Search Hyperparameters
    best_search_acc = 0
    best_beta, best_alpha, best_beta2, best_alpha2, best_lam = 0, 0, 0, 0, 0
    beta_list = [i * (cfg['search_scale2'][0] - cfg['search_scale'][0]) / cfg['search_step'][0] + cfg['search_scale'][0] for i in range(cfg['search_step'][0])]
    alpha_list = [i * (cfg['search_scale2'][1] - cfg['search_scale'][1]) / cfg['search_step'][1] + cfg['search_scale'][1] for i in range(cfg['search_step'][1])]
    # beta2_list = [i * (cfg['search_scale2'][2] - cfg['search_scale'][2]) / cfg['search_step'][2] + cfg['search_scale'][2] for i in range(cfg['search_step'][2])]
    beta2_list = [0]
    alpha2_list = [0]
    lam_list = [i * (cfg['search_scale2'][4] - cfg['search_scale'][4]) / cfg['search_step'][4] + cfg['search_scale'][4] for i in range(cfg['search_step'][4])]
    for beta in beta_list:
        for alpha in alpha_list:
            for beta2 in [0]:
                for alpha2 in [0]:
                    for lam in lam_list:
                        with torch.no_grad():
                            new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_neutral, R_FW = positive_adapter(cache_keys, negative_cache_keys, clip_weights_template, clip_weights_cupl, clip_weights_neutral, cache_values)
                            new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_neutral, R_FW = negative_adapter(new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_neutral, R_FW)
            
                            # Positive
                            R_fF = val_features @ new_cache_keys.half().t()
                            Aff = ((-1) * (beta - beta * (R_fF))).exp()
                            cache_logits = Aff @ soft_cache_values
                            new_clip_weights = 0.45 * new_clip_weights_template + 0.55 * new_clip_weights_cupl
                            R_fW = 100. * (val_features @ new_clip_weights) 
                            
                            # Negative
                            R_fF2 = (1 - val_features @ new_negative_cache_keys.half().t())
                            Aff2 = ((-1) * (beta - beta * (R_fF2))).exp()
                            cache_logits2 = Aff2 @ cache_values
                            R_fW2 = 100. * (1 - val_features @ new_clip_weights_neutral) * 0.15
                        
                            ape_logits = R_fW + cache_logits * alpha
                            ape_logits2 = R_fW2 + cache_logits2 * alpha
                            final_logits = lam * ape_logits + (1 - lam) * ape_logits2
                            acc = cls_acc(final_logits, val_labels)
                
                            if acc > best_search_acc:
                                print("New best setting, alpha: {:.2f}, beta: {:.2f}, alpha2: {:.2f}, beta2: {:.2f}, lam: {:.2f}; accuracy: {:.2f}".format(alpha, beta, alpha2, beta2, lam, acc))
                                best_search_acc = acc
                                best_beta, best_alpha, best_beta2, best_alpha2, best_lam = beta, alpha, beta2, alpha2, lam
        print("Finish beta={:.2f}/{:.2f} ({:.2f}%).".format(beta, beta_list[-1], beta / beta_list[-1] * 100))
    print("\nAfter searching, the best val accuarcy: {:.2f}.\n".format(best_search_acc))
    
    
    print("\n-------- Evaluating on the test set. --------")
    with torch.no_grad():
        new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_neutral, R_FW = positive_adapter(cache_keys, negative_cache_keys, clip_weights_template, clip_weights_cupl, clip_weights_neutral, cache_values)
        new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_neutral, R_FW = negative_adapter(new_cache_keys, new_negative_cache_keys, new_clip_weights_template, new_clip_weights_cupl, new_clip_weights_neutral, R_FW)
        
        # Positive
        R_fF = test_features @ new_cache_keys.half().t()
        Aff = ((-1) * (best_beta - best_beta * (R_fF))).exp()
        cache_logits = Aff @ soft_cache_values
        new_clip_weights = 0.45 * new_clip_weights_template + 0.55 * new_clip_weights_cupl
        R_fW = 100. * (test_features @ new_clip_weights) 
        
        # Negative
        R_fF2 = (1 - test_features @ new_negative_cache_keys.half().t())
        Aff2 = ((-1) * (best_beta - best_beta * (R_fF2))).exp()
        cache_logits2 = Aff2 @ cache_values
        R_fW2 = 100. * (1 - test_features @ new_clip_weights_neutral) * 0.15
    
        ape_logits = R_fW + cache_logits * best_alpha
        ape_logits2 = R_fW2 + cache_logits2 * best_alpha
        final_logits = best_lam * ape_logits + (1 - best_lam) * ape_logits2
        acc = cls_acc(final_logits, test_labels)
    print("**** DualAdapter's final test accuracy: {:.2f}. ****\n".format(acc)) 


def adaptive_reranking_loss(
    visual_features: torch.Tensor,
    class_prototypes: torch.Tensor,
    labels: torch.Tensor,
    scale: float = 4.0,
    knn: int = 3,
    **_: torch.Tensor,
) -> torch.Tensor:

    N = visual_features.shape[0]
    C = class_prototypes.shape[0]
    knn = min(knn, C)
    
    
    distances = torch.cdist(visual_features.float(), class_prototypes.float(), p=2)

    sorted_distances, sorted_indices = torch.sort(
        distances, dim=1, descending=False)
    anchor = (
        ((visual_features - class_prototypes[labels]) ** 2).sum(-1).sqrt().unsqueeze(1)
    )
    sorted_distances = sorted_distances[:, :knn]

    pos_cla_proto = class_prototypes[labels].unsqueeze(1)
    all_cls = class_prototypes[sorted_indices[:, :knn]]
    margins = (1.0 - (all_cls * pos_cla_proto).sum(-1)) / scale

    loss = torch.max(
        anchor + margins - sorted_distances,
        torch.zeros(N, knn).to(visual_features.device),
    )

    return loss.mean()

def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['shots'] = args.shot
    print(cfg['shots'])

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    print(cfg)

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights_template, clip_weights_cupl, clip_weights_neutral = load_text_feature(cfg)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = load_few_shot_feature(cfg)

    test_dataset = args.test
    if test_dataset is None:
        # Pre-load val features
        print("\nLoading visual features and labels from val set.")
        val_features, val_labels = loda_val_test_feature(cfg, "val")
        
        # Pre-load test features
        print("\nLoading visual features and labels from test set.")
        if cfg['dataset'] == 'imagenet':
            test_features, test_labels = loda_val_test_feature(cfg, "val")
        else:
            test_features, test_labels = loda_val_test_feature(cfg, "test")
    else:
        cache_dir = os.path.join('./caches', test_dataset)
        val_features = torch.load(cache_dir + "/" + "val" + "_f.pt")
        val_labels = torch.load(cache_dir + "/" + "val" + "_l.pt")
        test_features = torch.load(cache_dir + "/" + "test" + "_f.pt")
        test_labels = torch.load(cache_dir + "/" + "test" + "_l.pt")

    if cfg['dataset'] == 'imagenet':
        imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)
        train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)
    else:   
        dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)
    DualAdapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights_template, clip_weights_cupl, clip_weights_neutral, clip_model, train_loader_F)

if __name__ == '__main__':
    main()
