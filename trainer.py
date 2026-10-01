# import os
# import time
# import datetime
# import numpy as np
# from tqdm import tqdm
# import warnings
# import torch
# import torch.nn as nn
# from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
# from torchvision import transforms
# #from lavis.models import load_model_and_preprocess
# import json
# from clip import clip
# from model import Model

# from utils.meter import AverageMeter
# from utils.torchtools import load_checkpoint, save_checkpoint, resume_from_checkpoint

# from datasets.data import DATASET_GETTERS
# from itertools import cycle

# from utils.logger_SSL import Logger
# import itertools
# from copy import deepcopy

# from utils.utils import *
# from utils.utils import _ECELoss

# from collections import Counter

# best_acc = 0
# best_zs_acc = 0
# best_acc1 = 0


# torch.multiprocessing.set_sharing_strategy('file_system')
# import torch
# import torch.nn.functional as F

# import torch
# import torch.nn.functional as F

# def graph_regularization_loss_memory_2(
#     features,             # [B, D] 当前 batch 特征
#     pseudo_labels,        # [B]
#     class_preference,     # [C]
#     confidence,           # [B]
#     memory_feats,         # [M, D] memory bank 特征 (no grad)
#     memory_labels,        # [M]
#     memory_conf,          # [M]
#     alpha=0.5,            # repel loss 权重
#     margin=1.0,           # repel margin
#     bias_eps=0.1,         # 偏好差异阈值
#     method='mean',        # 置信度融合方式: 'mean' | 'product' | 'exp'
#     conf_thresh=0.7       # 高置信度阈值
# ):
#     B = features.size(0)
#     M = memory_feats.size(0)
#     device = features.device
#     dtype = features.dtype

#     # --- Normalize ---
#     feat = F.normalize(features.to(dtype=memory_feats.dtype), dim=1)
#     mem_feat = F.normalize(memory_feats, dim=1)
#     dist = torch.cdist(feat, mem_feat, p=2).pow(2)  # [B, M]

#     # --- Confidence fusion ---
#     conf = confidence.unsqueeze(1)
#     mem_conf = memory_conf.unsqueeze(0)
#     if method == 'product':
#         conf_weight = torch.sqrt(conf * mem_conf)
#     elif method == 'mean':
#         conf_weight = 0.5 * (conf + mem_conf)
#     elif method == 'exp':
#         conf_weight = torch.exp((conf * mem_conf) - 1)
#     else:
#         raise ValueError("Unsupported confidence fusion strategy")

#     # --- High-confidence gating mask ---
#     mask_curr = (confidence > conf_thresh).float().unsqueeze(1)
#     mask_mem = (memory_conf > conf_thresh).float().unsqueeze(0)
#     gating = mask_curr * mask_mem
#     conf_weight = conf_weight * gating

#     # --- Class preference difference (normalized) ---
#     sample_pref = class_preference[pseudo_labels]
#     mem_pref = class_preference[memory_labels]
#     bias_diff = torch.abs(sample_pref.unsqueeze(1) - mem_pref.unsqueeze(0))
#     bias_diff = bias_diff / (bias_diff.max() + 1e-6)
#     repel_mask = (bias_diff > bias_eps).float()

#     # --- Pull: same-class pairs ---
#     bias_weight = torch.exp(1-sample_pref).unsqueeze(1)
#     same_class_mask = (pseudo_labels.unsqueeze(1) == memory_labels.unsqueeze(0)).float()
#     pull_weight = same_class_mask * conf_weight * bias_weight
#     pull_loss = (pull_weight * dist).sum() / same_class_mask.sum().clamp(min=1.0)

#     # --- Repel: preference-different pairs ---
#     repel_term = F.relu(margin - dist)
#     repel_loss = (bias_diff * conf_weight * repel_term * repel_mask).sum() / repel_mask.sum().clamp(min=1.0)
#     # print('/////')
#     # --- Total loss ---
#     total_loss = pull_loss + alpha * repel_loss
#     # print(bias_weight)
#     return total_loss


# def bias_aware_topk_distribution(query_feats, bank_feats, bank_labels, bank_logits,
#                                  bias_vector, K=10, alpha=1.0, temperature=0.07):
#     B, D = query_feats.shape
#     N, C = bank_logits.shape[0], bank_logits.shape[1]

#     if (bank_labels == -1).any():
#         return None
    
#     query_feats = F.normalize(query_feats, p=2, dim=1)   # [B, D]
#     bank_feats  = F.normalize(bank_feats.detach(),  p=2, dim=1)   # [N, D]

#     sim = torch.matmul(query_feats, bank_feats.T)  # [B, N]
#     topk_val, topk_idx = torch.topk(sim, K, dim=-1)    # [B, K]
#     neigh_logits = bank_logits[topk_idx]               # [B, K, C]
#     neigh_labels = bank_labels[topk_idx]               # [B, K]
#     neigh_probs  = F.softmax(neigh_logits, dim=-1)     # [B, K, C]

#     raw_weights = F.softmax(topk_val / temperature, dim=-1)  # [B, K]
#     bias_penalty = 1.0 / (1.0 + alpha * bias_vector[neigh_labels])  # [B, K]
#     weights = raw_weights * bias_penalty
#     weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # [B, K]
#     weighted_prob = (neigh_probs * weights.unsqueeze(-1)).sum(dim=1)  # [B, C]

#     return weighted_prob


# def bias_aware_mhe_loss(classifier_weights, class_bias, gamma=1.0, beta=0.5, eps=1e-6):
#     """
#     classifier_weights: [C, D]
#     class_bias: [C]  # 可用类别频率 / 偏好分布（需归一化到 sum=1）
#     """
#     W = F.normalize(classifier_weights, dim=1)  # 单位化
#     cos = W @ W.t()
#     C = cos.size(0)
#     mask = ~torch.eye(C, device=W.device, dtype=torch.bool)
#     cos_ij = cos[mask]  # 取非对角元素

#     # pairwise bias权重
#     f = class_bias / (class_bias.sum() + eps)
#     # f = class_bias 
#     Fi, Fj = torch.meshgrid(f, f, indexing='ij')
#     bias_pair = (torch.max(Fi, Fj) / (torch.min(Fi, Fj) + eps)) ** beta
#     bias_pair = bias_pair[mask]

#     # MHE 斥力
#     base = (1.0 - cos_ij).clamp_min(eps) ** gamma
#     loss = (bias_pair / (base + eps)).mean()
#     return loss




# # def graph_regularization_loss(features, pseudo_labels, class_preference, confidence, alpha=1.0, tau=0.5, bias_eps=0.1):
# #     B = features.size(0)
# #     dtype, device = features.dtype, features.device
# #     feat = F.normalize(features, dim=1)
# #     dist = torch.cdist(feat, feat, p=2)
# #     sim = torch.exp(-dist / tau)
# #     conf = confidence.unsqueeze(1)
# #     conf_weight = 0.5 * (conf + conf.T)
# #     sample_pref = class_preference[pseudo_labels]
# #     bias_diff = (sample_pref.unsqueeze(1) - sample_pref.unsqueeze(0)).abs()
# #     same_mask = (pseudo_labels.unsqueeze(1) == pseudo_labels.unsqueeze(0)).to(dtype)
# #     same_mask *= (1 - torch.eye(B, dtype=dtype, device=device))
# #     repel_mask = (bias_diff > bias_eps).to(dtype)
# #     pull_loss = (conf_weight * same_mask * dist.pow(2)).sum() / same_mask.sum().clamp(min=1.0)
# #     margin = bias_diff * alpha
# #     repel_term = F.relu(margin - dist) ** 2
# #     repel_loss = (conf_weight * repel_mask * repel_term).sum() / repel_mask.sum().clamp(min=1.0)
# #     total_loss = pull_loss + alpha * repel_loss
# #     return total_loss


# # def graph_regularization_loss(
# #     features, pseudo_labels, class_preference, confidence,
# #     alpha=1.0, tau=0.5, bias_eps=0.1, margin_base=1.0, adaptive=True
# # ):
# #     B = features.size(0)
# #     dtype, device = features.dtype, features.device

# #     # --- Normalize features ---
# #     feat = F.normalize(features, dim=1)
# #     dist = torch.cdist(feat, feat, p=2)  # Euclidean distance

# #     # --- Compute bias & confidence weights ---
# #     sample_pref = class_preference[pseudo_labels]                # [B]
# #     bias_diff = (sample_pref.unsqueeze(1) - sample_pref.unsqueeze(0)).abs()  # [B,B]

# #     same_mask = (pseudo_labels.unsqueeze(1) == pseudo_labels.unsqueeze(0)).to(dtype)
# #     same_mask *= (1 - torch.eye(B, dtype=dtype, device=device))  # remove diag
# #     repel_mask = (bias_diff > bias_eps).to(dtype)

# #     conf = confidence.unsqueeze(1)
# #     conf_weight = 0.5 * (conf + conf.T)  # [B,B]
# #     # --- Split into pull/repel parts ---
# #     # high-confidence encourages pull, low-confidence enhances repel
# #     conf_pull = conf_weight
# #     conf_repel = (1 - conf_weight)

# #     # --- Pull loss (soft) ---
# #     pull_term = conf_pull * same_mask * dist.pow(2)
# #     pull_loss = pull_term.sum() / (same_mask.sum().clamp(min=1.0))

# #     # --- Adaptive margin based on bias and distance statistics ---
# #     if adaptive:
# #         # margin shrinks as mean distance grows (stabilization)
# #         mean_dist = dist.mean().detach()
# #         margin = margin_base * (bias_diff + 0.2) / (1 + mean_dist)
# #     else:
# #         margin = margin_base * bias_diff

# #     repel_term = F.relu(margin - dist) ** 2
# #     repel_loss = (conf_repel * repel_mask * repel_term).sum() / repel_mask.sum().clamp(min=1.0)

# #     # --- Combine ---
# #     total_loss = pull_loss + alpha * repel_loss
# #     return total_loss

# def graph_regularization_loss(
#     features,             # [B, D]
#     pseudo_labels,        # [B]
#     class_preference,     # [C]
#     confidence,           # [B]
#     alpha=0.5,           # repel loss weight
#     margin=1.0,           # margin for repel
#     bias_eps=0.1,         # preference difference threshold
#     method='mean',        # confidence fusion: 'mean' | 'product' | 'exp'
#     conf_thresh=0.7       # threshold for high-confidence gating
# ):
#     B = features.size(0)
#     # --- Cosine distance ---
#     # feat = F.normalize(features, dim=1)
#     # cos_sim = feat @ feat.T                      # [B, B], cosine similarity
#     # dist = 1.0 - cos_sim                         # [B, B], cosine distance ∈ [0, 2]

#     feat = F.normalize(features, dim=1)
#     dist = torch.cdist(feat, feat, p=2).pow(2)  # [B, B]

#     # --- Confidence weight ---
#     conf = confidence.unsqueeze(1)  # [B, 1]
#     if method == 'product':
#         conf_weight = torch.sqrt(conf @ conf.T)
#     elif method == 'mean':
#         conf_weight = 0.5 * (conf + conf.T)
#     elif method == 'exp':
#         conf_weight = torch.exp((conf @ conf.T) - 1)
#     else:
#         raise ValueError("Unsupported confidence fusion strategy")

#     # --- High-confidence gating mask ---
#     mask = (confidence > conf_thresh).float().unsqueeze(1)  # [B, 1]
#     gating = mask @ mask.T  # [B, B]
#     conf_weight = conf_weight * gating

#     # --- Class preference difference (normalized) ---
#     sample_pref = class_preference[pseudo_labels]  # [B]
#     bias_diff = torch.abs(sample_pref.unsqueeze(1) - sample_pref.unsqueeze(0))  # [B, B]
#     bias_diff = bias_diff / (bias_diff.max() + 1e-6)  # normalize to [0, 1]
#     repel_mask = (bias_diff > bias_eps).float()

#     # --- Pull: same-class pairs ---

    
#     bias_weight = torch.exp(1-sample_pref)
   
#     same_class_mask = (pseudo_labels.unsqueeze(1) == pseudo_labels.unsqueeze(0)).float()
#     same_class_mask = same_class_mask * (1 - torch.eye(B, device=feat.device))  # remove self-pairs
#     pull_weight = same_class_mask * conf_weight * bias_weight
#     pull_loss = (pull_weight * dist).sum() / same_class_mask.sum().clamp(min=1.0)
    
#     repel_term = F.relu(margin - dist)
#     repel_loss = (bias_diff * conf_weight * repel_term * repel_mask).sum() / repel_mask.sum().clamp(min=1.0)
#     total_loss = pull_loss + alpha * repel_loss
#     return total_loss


# def load_clip_to_cpu(cfg):
#     backbone_name = cfg.backbone
#     url = clip._MODELS[backbone_name]
#     model_path = clip._download(url)

#     try:
#         # loading JIT archive
#         model = torch.jit.load(model_path, map_location="cpu").eval()
#         state_dict = None

#     except RuntimeError:
#         state_dict = torch.load(model_path, map_location="cpu")

#     model = clip.build_model(state_dict or model.state_dict())
#     # print(model)
#     return model


# def forward_reg_loss(pred_logits):
#     pred_softmax = F.softmax(pred_logits, dim=1)
#     ent_loss = - (pred_softmax * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()
#     prob_mean = pred_softmax.mean(dim=0)
#     ne_loss = (prob_mean * prob_mean.log()).sum()
#     return ent_loss + ne_loss


# def compute_adjustment_by_py(py, tro, device):
#     adjustments = torch.log(py ** tro + 1e-12)
#     adjustments = adjustments.to(device)
#     return adjustments


# def interleave(x, size):
#     s = list(x.shape)
#     return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# def label_smoothing(logits, labels, features, class_templates, sigma=0.1):
#     batch_size, N = logits.shape
#     tau = 0.1
#     features = F.normalize(features, dim=1)
#     class_templates = F.normalize(class_templates, dim=1)
#     cos_sim = torch.matmul(features, class_templates.T) / 0.5  # [batch_size, N]
#     # cos_sim = torch.matmul(features, class_templates.T)  # [batch_size, N]
#     labels_one_hot = F.one_hot(labels, num_classes=N).float().to(logits.device)  # [batch_size, N]
#     masked_cos_sim = cos_sim.masked_fill(labels_one_hot.bool(), float('-inf'))
#     softmax_sim = F.softmax(masked_cos_sim, dim=1)
#     smoothed_labels = sigma * softmax_sim
#     smoothed_labels.scatter_(1, labels.unsqueeze(1), 1 - sigma)
#     # print(torch.max(smoothed_labels, dim=1))
#     # print(torch.sum(smoothed_labels, dim=1))
#     # print(smoothed_labels)
#     return smoothed_labels


# def de_interleave(x, size):
#     s = list(x.shape)
#     return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# def text_embedding(filename, device):
#     all_text_features = torch.load(filename)

#     all_means = []
#     all_covs = []
#     for c in range(all_text_features.size(0)):
#         cls_text_features = all_text_features[c].cpu().numpy()
#         mean = np.mean(cls_text_features, axis=0)  # 1024
#         cov = np.cov(cls_text_features.T)  # 1024 x 1024
#         mean = torch.from_numpy(mean)
#         cov = torch.from_numpy(cov)
#         cov = cov.diag()
#         all_means.append(mean)
#         all_covs.append(cov)
#     means = torch.stack(tuple(all_means), dim=0).half().to(device)
#     covs = torch.stack(tuple(all_covs), dim=0).half().to(device)

#     return means, covs


# def proto(means, query):
#     query = F.normalize(query, dim=1)
#     T = 0.07
#     means = F.normalize(means, dim=1)
#     query_mean = query.mm(means.permute(1, 0))  # N*K
#     logit = query_mean / T
#     # logit = torch.softmax(logit, dim=1)
#     return logit

# def top5_accuracy_with_mask(gt_labels, predicted_probs, mask):
#     # Get the indices of the top-5 predicted probabilities for each sample
#     top5_preds = torch.topk(predicted_probs, k=20, dim=1, largest=False).indices
#     # Check if the ground truth labels are within the top-5 predictions
#     matches = torch.any(top5_preds == gt_labels.unsqueeze(1), dim=1)
#     # Apply the mask to ignore certain samples
#     masked_matches = matches * mask
#     # Calculate the accuracy for non-masked samples
#     if masked_matches.numel() == 0:
#         return 0.0  # Handle edge case: all samples are masked
#     # top5_accuracy = torch.sum(masked_matches.float())
#     return masked_matches



# def entropy_confidence(probs, eps=1e-4):
#     device = probs.device
#     dtype = probs.dtype
#     C = probs.size(1)
#     probs = probs.clamp_min(eps)
#     entropy = -(probs * probs.log()).sum(dim=-1)
#     logC = torch.log(torch.tensor(C, device=device, dtype=dtype))
#     confidence = 1.0 - entropy / logC
#     confidence = confidence.clamp(0.0, 1.0)
#     return confidence



# class Trainer:
#     def __init__(self, cfg):

#         if not torch.cuda.is_available():
#             self.device = torch.device("cpu")
#         elif cfg.gpu is None:
#             self.device = torch.device("cuda")
#         else:
#             torch.cuda.set_device(cfg.gpu)
#             self.device = torch.device("cuda:{}".format(cfg.gpu))

#         # Save as attributes some frequently used variables
#         self.start_epoch = self.epoch = 0
#         self.num_epochs = cfg.num_epochs
#         self.output_dir = cfg.output_dir

#         self.cfg = cfg
#         self.build_data_loader()
#         self.build_model()
#         # self.evaluator = Evaluator(cfg, self.cls_num_list)
#         self.best_result = -np.inf
#         self._writer = None
#         self.th = cfg.th
        
#         class_list = []
#         for i in range(cfg.DATA.NUMBER_CLASSES):
#             class_list.append(str(i))

#         title = 'PEL-SSL-' + cfg.DATA.NAME
#         self.logger = Logger(os.path.join(cfg.output_dir, cfg.text_name), title=title)
#         self.logger.set_names(['Top1 acc', 'Best Top1 acc', 'epoch'])

#     def build_data_loader(self):
#         cfg = self.cfg
#         labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[cfg.DATA.NAME](
#             cfg)

#         self.num_classes = cfg.DATA.NUMBER_CLASSES
#         self.classnames = labeled_dataset.classes

#         # self.sampled_cls_num_list = self.cls_num_list
#         self.train_label_loader = DataLoader(labeled_dataset, num_workers=cfg.DATA.NUM_WORKERS,
#                                              batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, drop_last=True,
#                                              pin_memory=True)

#         self.train_unlabel_loader = DataLoader(unlabeled_dataset, num_workers=cfg.DATA.NUM_WORKERS,
#                                                batch_size=cfg.DATA.BATCH_SIZE * self.cfg.DATA.MU_U, shuffle=True,
#                                                drop_last=True, pin_memory=True)

#         self.test_loader = DataLoader(
#             test_dataset,
#             num_workers=cfg.DATA.NUM_WORKERS,
#             sampler=SequentialSampler(test_dataset),
#             batch_size=100)

#     def build_model(self):
#         cfg = self.cfg
#         # classnames = self.classnames

#         print(f"Loading CLIP (backbone: {cfg.backbone})")
#         clip_model = load_clip_to_cpu(cfg)
#         clip_model.to(self.device)

#         print(cfg.prec)

#         assert cfg.prec in ["fp16", "fp32", "amp"]
#         if cfg.prec == "fp32" or cfg.prec == "amp":
#             # CLIP's default precision is fp16
#             clip_model.float()

#         if cfg.template is not None:
#             temp = cfg.template
#         else:
#             temp = "a photo of a {}."
#         print(temp)
#         print(self.classnames)
#         prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
#         # prompts = [c for c in self.classnames]
#         prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         prompts = prompts.to(self.device)

#         # data_root = '/mnt/raid1/zhengna/CVPR_2026/datasets'
#         # # text_data = 
#         # text_data_path = os.path.join(data_root, self.cfg.dataset+'.json')
#         # text_data = json.load(open(text_data_path,'r'))
#         # text_data_list = []
#         # keys = text_data.keys()
        
        

#         # prompts = []
#         # for c in self.classnames:
#         #     name = c.replace("_"," ")
#         #     if name in text_data:
#         #         texts = text_data[name]
#         #         prompts.extend(texts) 
#         #     else:
#         #         print(f"{name} not found")
#         # prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts])
#         # prompts = prompts.to(self.device)
        
#         # print(text_data_list)
#         # text_data_list = torch.cat()
#         with torch.no_grad():
#             text_features = clip_model.encode_text(prompts)
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#         num_classes = len(self.classnames)
#         num_prompts = text_features.shape[0] // num_classes
#         text_features = text_features.view(num_classes, num_prompts, -1)
#         text_features = text_features.mean(dim=1)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#         self.text_features = text_features
#         print(self.text_features.shape)
#         print("Building model")
#         self.model = Model(cfg, clip_model, self.text_features)
#         self.tuner = self.model.tuner
#         self.clip_model = clip_model
#         self.dtype = clip_model.dtype

#         print("Turning off gradients in the model")
#         for name, param in self.model.named_parameters():
#             param.requires_grad_(False)

#         for name, param in self.tuner.named_parameters():
#             param.requires_grad_(True)

#         total_params = sum(p.numel() for p in self.model.parameters())
#         print(f'Total params: {total_params}')
#         tuned_params = sum(p.numel() for p in self.tuner.parameters())
#         print(f'Tuned params: {tuned_params}')
#         head_params = sum(p.numel() for p in self.tuner.head.parameters())
#         tuned_params_without_head = tuned_params - head_params
#         print(f'Tuned params (w/o head): {tuned_params_without_head}')

#         self.optim = torch.optim.SGD(self.tuner.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
#                                      momentum=cfg.momentum)
#         self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, float(cfg.num_epochs))
#         self.scaler = GradScaler() if cfg.prec == "amp" else None

#         device_count = torch.cuda.device_count()
#         if device_count > 1 and cfg.gpu is None:
#             print(cfg.gpu)
#             print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
#             self.model = nn.DataParallel(self.model)
#         self.model.to(self.device)



#     def train(self):
#         global best_acc
#         global best_zs_acc
#         global best_acc1
#         # added for threshold filter
#         filtered = torch.tensor([]).to(device=self.device)
#         # added for threshold filter
#         filtered_batch = torch.tensor([]).to(self.device)
#         pseudo_label_acc = torch.tensor([]).to(self.device)
#         unselected_1 = torch.tensor([]).to(self.device)
#         unselected_2 = torch.tensor([]).to(self.device)

#         if self.cfg.resume:
#             directory = self.cfg.resume
#             self.start_epoch = self.resume_model_if_exist(directory)

#         self.tuner.head1.weight.data = self.tuner.head.weight.data.clone()
#         # Initialize summary writer
#         writer_dir = os.path.join(self.output_dir, "tensorboard")
#         os.makedirs(writer_dir, exist_ok=True)
#         print(f"Initialize tensorboard (log_dir={writer_dir})")
#         self._writer = SummaryWriter(log_dir=writer_dir)

#         self.w_con = self.cfg.w_con
#         ulab_len = len(self.train_unlabel_loader.dataset)

#         self.alpha = self.cfg.alpha
#         self.betabase = torch.ones((self.num_classes, self.num_classes)).to(self.device)
#         self.betabase[torch.arange(self.num_classes), torch.arange(self.num_classes)] = 0.0

#         self.smoothing = self.cfg.smoothing
#         self.th_min = self.cfg.th_min

#         self.time_start = time.time()

#         names = {'clip': '_clipL.pt', 'bert': '_BertL.pt', 'mT5': '_mT5L.pt'}
#         filename = self.cfg.dataset + names[self.cfg.text_model_name]
#         print('using text model: ', filename)
        
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", category=FutureWarning)
            
#         total_steps = self.cfg.total_steps * self.num_epochs
#         self.warmup_iters = int(0.25 * total_steps)
#         self.model.compute_static_bias(self.train_unlabel_loader, num_batches=250)
#         for self.epoch in range(self.start_epoch, self.num_epochs):
            
#             self.tuner.train()

#             batch_time = AverageMeter()
#             data_time = AverageMeter()
#             loss_meter = AverageMeter()
#             acc_meter = AverageMeter()
#             # added for threshold filter
#             filtered_epoch = torch.tensor([]).to(device=self.device)
#             epoch_acc = torch.tensor([]).to(self.device)
#             unselect_acc_1 = torch.tensor([]).to(self.device)
#             unselect_acc_2 = torch.tensor([]).to(self.device)
#             filter = torch.tensor([]).to(self.device)
#             self.num_batches = self.cfg.total_steps
#             label_loader_iter = cycle(self.train_label_loader)
#             unlabel_loader_iter = cycle(self.train_unlabel_loader)

#             selected_label = torch.ones((ulab_len,), dtype=torch.long, ) * -1
#             selected_label = selected_label.to(self.device)
#             classwise_acc = torch.zeros((self.num_classes,)).to(self.device)
#             end = time.time()

#             for self.batch_idx in range(self.cfg.total_steps):
                
#                 data_time.update(time.time() - end)

#                 (inputs_x, targets_x, _) = next(label_loader_iter)
#                 batch_size = inputs_x.shape[0]
#                 ((inputs_u_w, inputs_u_s, inputs_u_s1), u_real, uidx) = next(unlabel_loader_iter)

#                 uidx = uidx.to(self.device)
#                 targets_x = targets_x.to(self.device)
#                 targets_x = targets_x.to(torch.long)
#                 pseudo_counter = Counter(selected_label.tolist())

#                 inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1 = inputs_x.to(self.device), inputs_u_w.to(self.device), \
#                     inputs_u_s.to(self.device), inputs_u_s1.to(self.device)

#                 inputs = interleave(
#                     torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1)), 3 * self.cfg.DATA.MU_U + 1)
#                 feat = self.model(inputs)
#                 feat = de_interleave(feat, 3 * self.cfg.DATA.MU_U + 1)
#                 feat_l_w = feat[:batch_size]
#                 feat_u_w, feat_u_s, feat_u_s1 = feat[batch_size:].chunk(3)
#                 output = self.tuner.head(feat)
#                 output_x = output[:batch_size]
#                 output_u_w, output_u_s, output_u_s1 = output[batch_size:].chunk(3)
#                 # output1 = self.tuner.head1(feat.detach())
#                 # output1_x = output1[:batch_size]
#                 # output1_u_w, output1_u_s, output1_u_s1 = output1[batch_size:].chunk(3)
#                 del feat
#                 del output
#                 # del output1

#                 pseu = torch.softmax(output_u_w.detach(), dim=-1)
#                 conf, targets_u = torch.max(pseu, dim=-1)
#                 mask = conf.ge(self.th)
#                 mask_twice = torch.cat([mask, mask], dim=0).to(self.device)
#                 mask_twice = mask_twice.half()
#                 output_u_s_twice = torch.cat([output_u_s, output_u_s1], dim=0).to(self.device)
#                 targets_u_twice = torch.cat([targets_u, targets_u], dim=0).to(self.device)
#                 L_reg = forward_reg_loss(output_u_s_twice) + forward_reg_loss(output_u_w)
                
                
#                 bias, logit_adjust = self.model.masking_1(pseu, feat_u_w, softmax_x_ulb=False, factor=True)
#                 bias_pred = bias[targets_u] 

#                 # alpha_eff = (1 - conf)**2
#                 # ent_conf = entropy_confidence(pseu)
#                 # confidence = (ent_conf ** alpha_eff) * (conf ** (1 - alpha_eff))
#                 self.cur_bias = bias
#                 alpha = self.cfg.alpha
#                 sigma = self.cfg.sigma
#                 lamda = self.cfg.lamda
#                 tem = self.cfg.tem
#                 # print(alpha,'--------------------')
#                 ent_conf = entropy_confidence(pseu)
#                 # confidence = (ent_conf ** 0.1) * (conf ** (1 - 0.1))
#                 confidence  = (conf** 1) * (ent_conf ** 0)
#                 # print('----------')
#                 # print(confidence, conf)
#                 scores = confidence #* torch.exp((1-bias_pred) * tem)
#                 scores_twice = torch.cat([scores,scores],dim=0)
#                 Lx = F.cross_entropy(output_x, targets_x, reduction='mean')
#                 self.model.update_ema_encoder()
#                 feats = self.model.enqueue_from_ema(inputs_u_w, output_u_w, feat_u_w)
#                 feats = F.normalize(feats, dim=1)
#                 d_j = self.model.compute_relative_distance(feats.detach(), targets_u)
#                 r_bias = torch.exp((1 - bias_pred) * d_j.to(bias_pred.dtype))
#                 r_bias_twice = torch.cat([r_bias, r_bias], dim=0)
#                 # self.model.enqueue_from_logits(feat_u_w, output_u_w)
#                 mask_ent = confidence.ge(self.th)
#                 mask_ent_twice = torch.cat([mask_ent, mask_ent], dim=0).to(self.device)
#                 mask_ent_twice = mask_ent_twice.half()
                

#                 if torch.sum(mask_twice) > 0:
#                     # Lu = (F.cross_entropy(output_u_s_twice+logit_adjust, targets_u_twice,
#                     #                       reduction='none')* scores_twice * mask_twice).mean()
#                     Lu = (F.cross_entropy(output_u_s_twice+logit_adjust, targets_u_twice,
#                                           reduction='none') * scores_twice* mask_twice).mean()
#                     # Lu = (F.cross_entropy(output_u_s_twice+logit_adjust, targets_u_twice,
#                     #                       reduction='none')  * mask_twice).mean()
#                     # Lu = (F.cross_entropy(output_u_s_twice+logit_adjust, targets_u_twice,
#                     #                       reduction='none') * scores_twice * mask_twice).mean()
#                     # graph_loss = graph_regularization_loss(features=feat_u_w, pseudo_labels=targets_u, class_preference=bias, confidence=confidence.detach()) #+ \
                
#                     queue, queue_labels, queue_logits = self.model.get_queue()
#                     queue_prob = torch.softmax(queue_logits, dim=1)  # [M, C]
#                     queue_conf, _ = queue_prob.max(dim=1) 
#                     entropy_conf = entropy_confidence(queue_prob)
#                     queue_confidence = (entropy_conf ** alpha) * (queue_conf ** (1 - alpha))
#                     graph_loss = graph_regularization_loss_memory_2(features=feat_u_w,pseudo_labels=targets_u,class_preference=bias,confidence=confidence.detach(), memory_feats=queue,memory_labels=queue_labels, memory_conf=queue_conf.detach(), alpha = sigma)
                    
                
#                 else:
#                     Lu = 0
#                     graph_loss = 0
                
#                 loss = Lx + Lu * self.w_con + graph_loss * lamda * self.model.time_p +  0.1 * L_reg #+ graph_loss * lamda * self.model.time_p
                
#                 self.optim.zero_grad()
#                 loss.backward()
#                 self.optim.step()

#                 epoch_acc = torch.cat((epoch_acc, (targets_u == u_real.to(self.device)).float()), dim=0)
               
#                 filtered_batch = torch.cat((filtered_batch, mask_twice.float().mean(-1, keepdim=True)), dim=0)
#                 filtered_epoch = torch.cat((filtered_epoch, mask_twice.float()), dim=0)
                
#                 current_logit_scale = self.tuner.head.logit_scale.item()
#                 current_lr = self.optim.param_groups[0]["lr"]
#                 loss_meter.update(loss.item())

#                 batch_time.update(time.time() - end)

#                 meet_freq = (self.batch_idx + 1) % self.cfg.print_freq == 0
#                 only_few_batches = self.num_batches < self.cfg.print_freq
#                 if meet_freq or only_few_batches:
#                     nb_remain = 0
#                     nb_remain += self.num_batches - self.batch_idx - 1
#                     nb_remain += (
#                                          self.num_epochs - self.epoch - 1
#                                  ) * self.num_batches
#                     eta_seconds = batch_time.avg * nb_remain
#                     eta = str(datetime.timedelta(seconds=int(eta_seconds)))

#                     info = []
#                     info += [f"epoch [{self.epoch + 1}/{self.num_epochs}]"]
#                     info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
#                     info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
#                     info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
#                     info += [f"s {current_logit_scale:.4f}"]
#                     info += [f"lr {current_lr:.4e}"]
#                     info += [f"eta {eta}"]
#                     print(" ".join(info))

#                 n_iter = self.epoch * self.num_batches + self.batch_idx
#                 self._writer.add_scalar("train/loss", loss_meter.avg, n_iter)
#                 self._writer.add_scalar("train/acc", acc_meter.avg, n_iter)
#                 self._writer.add_scalar("train/lr", current_lr, n_iter)

#                 if (self.batch_idx + 1) == self.num_batches:
#                     self.sched.step()

#                 end = time.time()

#             last_epoch = (self.epoch + 1) == self.num_epochs
#             meet_checkpoint_freq = (
#                 (self.epoch + 1) % self.cfg.checkpoint_freq == 0
#                 if self.cfg.checkpoint_freq > 0 else False
#             )

#             if meet_checkpoint_freq or last_epoch:
#                 self.save_model(self.epoch, self.output_dir)

#             # added for threshold filter
#             print(f"{filtered_epoch.min()=} {filtered_epoch.max()=} {filtered_epoch.mean()=} {filtered_epoch.std()=} {filtered_epoch.median()=}")
#             filtered = torch.cat((filtered, filtered_epoch.mean(-1, keepdim=True)), dim=-1)
#             pseudo_label_acc = torch.cat((pseudo_label_acc, epoch_acc.mean(-1, keepdim=True)), dim=-1)
#             print(pseudo_label_acc)

#             unselect_acc_1 = torch.sum(unselect_acc_1) / torch.sum(filter)
#             unselect_acc_2 = torch.sum(unselect_acc_2) / torch.sum(filter)
#             unselect_acc_1 = unselect_acc_1.unsqueeze(0)
#             unselect_acc_2 = unselect_acc_2.unsqueeze(0)
#             unselected_1 = torch.cat([unselected_1, unselect_acc_1], dim=-1)
#             unselected_2 = torch.cat([unselected_2, unselect_acc_2], dim=-1)

#             acc_now = self.test()
#             best_acc = max(best_acc, acc_now)
#             self.logger.append([acc_now, best_acc, self.epoch + 1])
#             # if self.epoch % 5 == 0 and self.epoch != 0:
#             #     self.save_test_snapshot(self.epoch)
#             # elif last_epoch:
#             #     self.save_test_snapshot(self.epoch)


        
#         print("Finish training")
#         # added for threshold filter
#         # assert filtered.shape[0] == self.cfg.num_epochs
#         torch.save(dict(epoch=filtered.cpu(),
#                         batch=filtered_batch.cpu(),
#                         pseudo_acc=pseudo_label_acc.cpu()),
#                    self.cfg.output_dir + '/pseudo_acc.pt')


#         print("Deploy the last-epoch model for testing")
#         # self.save_epoch_snapshot(
#         # Show elapsed time
#         elapsed = round(time.time() - self.time_start)
#         elapsed = str(datetime.timedelta(seconds=elapsed))
#         print(f"Elapsed: {elapsed}")

#         # Close writer
#         self._writer.close()

#         self.logger.close()

#     @torch.no_grad()
#     def test(self):
#         self.tuner.eval()
#         print(f"Evaluate on the test set")

#         preds = np.array([])
#         targets = np.array([])
#         feats = []
#         labels = torch.tensor([], dtype=torch.long)
#         outputs = []

#         for batch in tqdm(self.test_loader, ascii=True):
#             image = batch[0].to(self.device, non_blocking=True)
#             label = batch[1].to(self.device, non_blocking=True)

#             feat = self.model(image)
#             output = self.tuner.head(feat)

#             feats.append(feat.cpu())

#             prob = F.softmax(output, dim=1)
#             conf, pred = prob.max(1)
#             preds = np.append(preds, pred.cpu().numpy())
#             targets = np.append(targets, label.cpu().numpy())

#             outputs.append(output.cpu())
#             labels = torch.cat([labels, label.cpu()], dim=-1)

#         targets = targets.astype(int)
#         preds = preds.astype(int)

#         outputs = torch.cat(outputs, dim=0)
#         acc = sum(targets == preds) / len(targets)

#         # ── per-class accuracy ────────────────────────────────────────
#         targets_t = torch.tensor(targets)
#         preds_t   = torch.tensor(preds)
#         correct   = torch.zeros(self.model.class_num)
#         total     = torch.zeros(self.model.class_num)
#         for c in range(self.model.class_num):
#             mask       = (targets_t == c)
#             correct[c] = (preds_t[mask] == c).sum()
#             total[c]   = mask.sum()
#         per_class_acc = correct / (total + 1e-6)

#         # 打出 scatter 和 label_hist 对应类的 accuracy
#         scatter_norm  = self.model.compute_scatter()
#         hist          = self.model.label_hist

#         scatter_top5  = scatter_norm.cpu().topk(5).indices
#         scatter_bot5  = scatter_norm.cpu().topk(5, largest=False).indices
#         hist_top5     = hist.cpu().topk(5).indices

#         print(f"scatter top5 类: {scatter_top5.tolist()} → acc: {per_class_acc[scatter_top5].tolist()}")
#         print(f"scatter bot5 类: {scatter_bot5.tolist()} → acc: {per_class_acc[scatter_bot5].tolist()}")
#         print(f"label_hist top5 类: {hist_top5.tolist()} → acc: {per_class_acc[hist_top5].tolist()}")
#         print(f"全局平均 accuracy: {per_class_acc.mean():.4f}")
#         # ─────────────────────────────────────────────────────────────

#         feats_tensor = torch.cat(feats, dim=0)
#         return acc
#     # def test(self):
#     #     self.tuner.eval()
#     #     print(f"Evaluate on the test set")

#     #     preds = np.array([])
#     #     targets = np.array([])
#     #     feats = []
#     #     labels = torch.tensor([], dtype=torch.long)
#     #     outputs = []

#     #     for batch in tqdm(self.test_loader, ascii=True):
#     #         image = batch[0].to(self.device, non_blocking=True)
#     #         label = batch[1].to(self.device, non_blocking=True)

#     #         feat = self.model(image)
#     #         output = self.tuner.head(feat)

#     #         feats.append(feat.cpu())

#     #         prob = F.softmax(output, dim=1)
#     #         conf, pred = prob.max(1)
#     #         preds = np.append(preds, pred.cpu().numpy())
#     #         targets = np.append(targets, label.cpu().numpy())

#     #         outputs.append(output.cpu())
#     #         labels = torch.cat([labels, label.cpu()], dim=-1)

#     #     targets = targets.astype(int)
#     #     preds = preds.astype(int)

#     #     outputs = torch.cat(outputs, dim=0)
#     #     acc = sum(targets == preds) / len(targets)
#     #     print(outputs.shape)

#     #     feats_tensor = torch.cat(feats, dim=0)
        

#     #     # print(feats_tensor.shape)
#     #     return acc
#     def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
#         tuner_dict = self.tuner.state_dict()
#         optim_dict = self.optim.state_dict()
#         sched_dict = self.sched.state_dict()
#         save_checkpoint(
#             {
#                 "state_dict": tuner_dict,
#                 "epoch": epoch + 1,
#                 "optimizer": optim_dict,
#                 "scheduler": sched_dict,
#                 "val_result": val_result
#             },
#             os.path.join(directory + f'/epoch_{epoch + 1}', "tuner"),
#             is_best=is_best,
#             model_name=model_name,
#         )

#     def resume_model_if_exist(self, directory):
#         file_missing = False

#         path = os.path.join(directory, "tuner")
#         if not os.path.exists(path):
#             print("No checkpoint found, train from scratch")
#             return 0

#         print(f"Found checkpoint at {directory} (will resume training)")

#         path = os.path.join(directory, "tuner")
#         start_epoch = resume_from_checkpoint(
#             path, self.tuner, self.optim, self.sched
#         )

#         return start_epoch

#     def load_model(self, directory, epoch=None):
#         if not directory:
#             print("Note that load_model() is skipped as no pretrained model is given")
#             return

#         # By default, the best model is loaded
#         model_file = "model-best.pth.tar"

#         if epoch is not None:
#             model_file = "model.pth.tar-" + str(epoch)

#         model_path = os.path.join(directory, "tuner", model_file)

#         if not os.path.exists(model_path):
#             raise FileNotFoundError('Model not found at "{}"'.format(model_path))

#         checkpoint = load_checkpoint(model_path, self.device)
#         state_dict = checkpoint["state_dict"]
#         epoch = checkpoint["epoch"]

#         # Ignore fixed token vectors
#         if "token_prefix" in state_dict:
#             del state_dict["token_prefix"]

#         if "token_suffix" in state_dict:
#             del state_dict["token_suffix"]

#         print("Loading weights to {} " 'from "{}" (epoch = {})'.format("tuner", model_path, epoch))
#         # set strict=False
#         self.tuner.load_state_dict(state_dict, strict=False)

#     @torch.no_grad()
#     def get_comparison_data(self):
#         # operate on which data? Unlabelled weak data.
#         self.tuner.eval()
#         print(f"Getting data for comparison plots")
#         preds = torch.tensor([])
#         targets = torch.tensor([])
#         confidences = torch.tensor([])
#         rect_confidences = torch.tensor([])
#         # probs = list()
#         prob = torch.tensor([])
#         outputs = []
#         feats = []
#         for batch in tqdm(self.train_unlabel_loader, ascii=True):
#             ((images, _, _), label, _) = batch
#             # images, label, _ = batch
#             # print(f"{images.shape=}")

#             images = images.to(self.device, non_blocking=True)
#             label = label.to(self.device, non_blocking=True)

#             feat = self.model(images)
#             output = self.tuner.head(feat)
#             # del feat

#             prob = F.softmax(output, dim=1)
#             conf, pred = prob.max(1)
#             confidences = torch.cat((confidences, conf.cpu()), dim=-1)
#             preds = torch.cat((preds, pred.cpu()), dim=-1)
#             targets = torch.cat((targets, label.cpu()), dim=-1)
#             outputs.append(output.cpu())
#             feats.append(feat.cpu())
#             res = (pred == label).cpu()
#             acc = (res).float().mean()
#             k = acc / conf.mean().cpu()
#             rect_conf = k * conf.cpu()
#             rect_confidences = torch.cat((rect_confidences, rect_conf), dim=-1)

#         feats = torch.cat(feats, dim=0)
#         outputs = torch.cat(outputs, dim=0)
#         # plot 1
#         class_indices, counts = torch.unique(preds, return_counts=True)
#         sorted_counts, count_indices = torch.sort(counts, descending=True)
#         class_indices = class_indices[count_indices]

#         # plot 2
#         average_confidence = confidences.mean()

#         # plot 3 - ece
#         ece_scorer = _ECELoss()
#         ece = ece_scorer(preds=preds, targets=targets, confs=rect_confidences)

#         # plot 4
#         true_rc = rect_confidences[targets == preds]
#         false_rc = rect_confidences[~(targets == preds)]


#     @torch.no_grad()
#     def save_test_snapshot(self, epoch, use_half=True):
#         """
#         Save a snapshot of the test set (features, logits, labels, confidence, entropy).
#         Used mainly for visualization or analysis on the test distribution.
#         """

#         # ---- Sanity check ----
#         assert hasattr(self, "test_loader"), "❌ self.test_loader not found. Please define self.test_loader first."
#         dataloader = self.test_loader

#         self.model.eval()
#         self.tuner.eval()

#         all_feats, all_logits = [], []
#         all_labels, all_pseudo = [], []
#         all_conf, all_entropy = [], []

#         for batch in tqdm(dataloader, desc=f"[Test] Saving snapshot (epoch {epoch})"):
#             # Unpack batch
#             images = batch[0].to(self.device, non_blocking=True)
#             labels = batch[1].to(self.device, non_blocking=True)

#             # ---- Forward ----
#             feats = self.model(images)                  # [B, D]
#             logits = self.tuner.head(feats)             # [B, C]
#             probs = F.softmax(logits, dim=1)
#             conf, pseudo = probs.max(dim=1)
#             ent_conf = entropy_confidence(probs)

#             # ---- Collect ----
#             all_feats.append(feats.cpu())
#             all_logits.append(logits.cpu())
#             all_labels.append(labels.cpu())
#             all_pseudo.append(pseudo.cpu())
#             all_conf.append(conf.cpu())
#             all_entropy.append(ent_conf.cpu())

#         # ---- Concatenate ----
#         feats = torch.cat(all_feats)
#         logits = torch.cat(all_logits)
#         labels = torch.cat(all_labels)
#         pseudo = torch.cat(all_pseudo)
#         confs = torch.cat(all_conf)
#         entropy_conf = torch.cat(all_entropy)

#         # ---- Save ----
#         save_dict = {
#             'epoch': epoch,
#             'features': feats.half() if use_half else feats,
#             'logits': logits.half() if use_half else logits,
#             'true_labels': labels,
#             'pseudo_labels': pseudo,
#             'confidence': confs,
#             'entropy_conf': entropy_conf
#         }

#         os.makedirs(self.output_dir, exist_ok=True)
#         save_path = os.path.join(self.output_dir, f"epoch_{epoch:03d}_snapshot_test_our_2.pt")
#         torch.save(save_dict, save_path)
#         print(f" Saved test snapshot → {save_path}")

#     @torch.no_grad()
#     def save_epoch_snapshot(self, epoch, dataloader=None):
       
#         if dataloader is None:
#             dataloader = self.train_unlabel_loader

#         all_feats, all_logits, all_labels, all_pseudo, all_conf, all_entropy = [], [], [], [], [], []

#         self.model.eval()
#         self.tuner.eval()

#         for batch in tqdm(dataloader, desc=f"[Epoch {epoch}] Saving snapshot"):
#             ((images, _, _), labels, _) = batch
#             images = images.to(self.device)
#             labels = labels.to(self.device)

#             # 提取特征与 logits
#             feats = self.model(images)                   # [B, D]
#             logits = self.tuner.head(feats)              # [B, C]
#             probs = F.softmax(logits, dim=1)
#             conf, pseudo = probs.max(dim=1)
#             ent_conf = entropy_confidence(probs)

#             all_feats.append(feats.cpu())
#             all_logits.append(logits.cpu())
#             all_labels.append(labels.cpu())
#             all_pseudo.append(pseudo.cpu())
#             all_conf.append(conf.cpu())
#             all_entropy.append(ent_conf.cpu())

        
#         feats = torch.cat(all_feats)
#         logits = torch.cat(all_logits)
#         labels = torch.cat(all_labels)
#         pseudo = torch.cat(all_pseudo)
#         confs = torch.cat(all_conf)
#         entropy_conf = torch.cat(all_entropy)

#         # bias, _ = self.model.masking(F.softmax(logits, dim=1), softmax_x_ulb=False, factor=True)
#         bias = self.cur_bias
#         bias = bias.cpu()
#         class_counts = torch.bincount(pseudo, minlength=self.num_classes).float()
#         class_freq = class_counts / class_counts.sum()

#         save_dict = {
#             'epoch': epoch,
#             'features': feats.half(),
#             'logits': logits.half(),
#             'true_labels': labels,
#             'pseudo_labels': pseudo,
#             'bias': bias,
#             'class_freq': class_freq,
#             'confidence': confs,
#             'entropy_conf': entropy_conf
#         }

#         save_path = os.path.join(self.output_dir, f"epoch_{epoch}_snapshot_our_2.pt")
#         torch.save(save_dict, save_path)
#         print(f" Saved epoch snapshot → {save_path}")



#     @torch.no_grad()
#     def get_feature(self):
#         # operate on which data? Unlabelled weak data.
#         preds = torch.tensor([])  
#         targets = torch.tensor([])  
#         confidences = torch.tensor([])  
#         rect_confidences = torch.tensor([])  
#         prob = torch.tensor([])  
#         outputs = []  
#         feats = []  # 用来存储特征

#         for batch in tqdm(self.train_unlabel_loader, ascii=True):
#             ((images, _, _), label, _) = batch
#             images = images.to(self.device, non_blocking=True)
#             label = label.to(self.device, non_blocking=True)

#             feat = self.model(images)  # 获取当前批次的特征
#             outputs.append(feat.cpu())
#             targets = torch.cat((targets, label.cpu()), dim=-1)
#             feats.append(feat.cpu())  # 将特征存储到列表中

#         feats = torch.cat(feats, dim=0)  # 将所有批次的特征连接起来

#         # 保存特征和标签
#         torch.save({'features': feats, 'labels': targets}, 'features_and_labels_epoch_{}.pt'.format(self.epoch))
#         print(f"Saved features and labels for epoch {self.epoch}")

import os
import time
import datetime
import numpy as np
from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torchvision import transforms
#from lavis.models import load_model_and_preprocess
import json
from clip import clip
from model import Model

from utils.meter import AverageMeter
from utils.torchtools import load_checkpoint, save_checkpoint, resume_from_checkpoint

from datasets.data import DATASET_GETTERS
from itertools import cycle

from utils.logger_SSL import Logger
import itertools
from copy import deepcopy

from utils.utils import *
from utils.utils import _ECELoss

from collections import Counter

best_acc = 0
best_zs_acc = 0
best_acc1 = 0


torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def graph_regularization_loss_memory_2(
    features,             # [B, D] 当前 batch 特征
    pseudo_labels,        # [B]
    class_preference,     # [C]
    confidence,           # [B]
    memory_feats,         # [M, D] memory bank 特征 (no grad)
    memory_labels,        # [M]
    memory_conf,          # [M]
    alpha=0.5,            # repel loss 权重
    margin=1.0,           # repel margin
    bias_eps=0.1,         # 偏好差异阈值
    method='mean',        # 置信度融合方式: 'mean' | 'product' | 'exp'
    conf_thresh=0.7       # 高置信度阈值
):
    B = features.size(0)
    M = memory_feats.size(0)
    device = features.device
    dtype = features.dtype

    # --- Normalize ---
    feat = F.normalize(features.to(dtype=memory_feats.dtype), dim=1)
    mem_feat = F.normalize(memory_feats, dim=1)
    dist = torch.cdist(feat, mem_feat, p=2).pow(2)  # [B, M]

    # --- Confidence fusion ---
    conf = confidence.unsqueeze(1)
    mem_conf = memory_conf.unsqueeze(0)
    if method == 'product':
        conf_weight = torch.sqrt(conf * mem_conf)
    elif method == 'mean':
        conf_weight = 0.5 * (conf + mem_conf)
    elif method == 'exp':
        conf_weight = torch.exp((conf * mem_conf) - 1)
    else:
        raise ValueError("Unsupported confidence fusion strategy")

    # --- High-confidence gating mask ---
    mask_curr = (confidence > conf_thresh).float().unsqueeze(1)
    mask_mem = (memory_conf > conf_thresh).float().unsqueeze(0)
    gating = mask_curr * mask_mem
    conf_weight = conf_weight * gating

    # --- Class preference difference (normalized) ---
    sample_pref = class_preference[pseudo_labels]
    mem_pref = class_preference[memory_labels]
    bias_diff = torch.abs(sample_pref.unsqueeze(1) - mem_pref.unsqueeze(0))
    bias_diff = bias_diff / (bias_diff.max() + 1e-6)
    repel_mask = (bias_diff > bias_eps).float()

    # --- Pull: same-class pairs ---
    bias_weight = torch.exp(1-sample_pref).unsqueeze(1)
    same_class_mask = (pseudo_labels.unsqueeze(1) == memory_labels.unsqueeze(0)).float()
    pull_weight = same_class_mask * conf_weight * bias_weight
    pull_loss = (pull_weight * dist).sum() / same_class_mask.sum().clamp(min=1.0)

    # --- Repel: preference-different pairs ---
    repel_term = F.relu(margin - dist)
    repel_loss = (bias_diff * conf_weight * repel_term * repel_mask).sum() / repel_mask.sum().clamp(min=1.0)
    # print('/////')
    # --- Total loss ---
    total_loss = pull_loss + alpha * repel_loss
    # print(bias_weight)
    return total_loss


def bias_aware_topk_distribution(query_feats, bank_feats, bank_labels, bank_logits,
                                 bias_vector, K=10, alpha=1.0, temperature=0.07):
    B, D = query_feats.shape
    N, C = bank_logits.shape[0], bank_logits.shape[1]

    if (bank_labels == -1).any():
        return None
    
    query_feats = F.normalize(query_feats, p=2, dim=1)   # [B, D]
    bank_feats  = F.normalize(bank_feats.detach(),  p=2, dim=1)   # [N, D]

    sim = torch.matmul(query_feats, bank_feats.T)  # [B, N]
    topk_val, topk_idx = torch.topk(sim, K, dim=-1)    # [B, K]
    neigh_logits = bank_logits[topk_idx]               # [B, K, C]
    neigh_labels = bank_labels[topk_idx]               # [B, K]
    neigh_probs  = F.softmax(neigh_logits, dim=-1)     # [B, K, C]

    raw_weights = F.softmax(topk_val / temperature, dim=-1)  # [B, K]
    bias_penalty = 1.0 / (1.0 + alpha * bias_vector[neigh_labels])  # [B, K]
    weights = raw_weights * bias_penalty
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # [B, K]
    weighted_prob = (neigh_probs * weights.unsqueeze(-1)).sum(dim=1)  # [B, C]

    return weighted_prob


def bias_aware_mhe_loss(classifier_weights, class_bias, gamma=1.0, beta=0.5, eps=1e-6):
    """
    classifier_weights: [C, D]
    class_bias: [C]  # 可用类别频率 / 偏好分布（需归一化到 sum=1）
    """
    W = F.normalize(classifier_weights, dim=1)  # 单位化
    cos = W @ W.t()
    C = cos.size(0)
    mask = ~torch.eye(C, device=W.device, dtype=torch.bool)
    cos_ij = cos[mask]  # 取非对角元素

    # pairwise bias权重
    f = class_bias / (class_bias.sum() + eps)
    # f = class_bias 
    Fi, Fj = torch.meshgrid(f, f, indexing='ij')
    bias_pair = (torch.max(Fi, Fj) / (torch.min(Fi, Fj) + eps)) ** beta
    bias_pair = bias_pair[mask]

    # MHE 斥力
    base = (1.0 - cos_ij).clamp_min(eps) ** gamma
    loss = (bias_pair / (base + eps)).mean()
    return loss




# def graph_regularization_loss(features, pseudo_labels, class_preference, confidence, alpha=1.0, tau=0.5, bias_eps=0.1):
#     B = features.size(0)
#     dtype, device = features.dtype, features.device
#     feat = F.normalize(features, dim=1)
#     dist = torch.cdist(feat, feat, p=2)
#     sim = torch.exp(-dist / tau)
#     conf = confidence.unsqueeze(1)
#     conf_weight = 0.5 * (conf + conf.T)
#     sample_pref = class_preference[pseudo_labels]
#     bias_diff = (sample_pref.unsqueeze(1) - sample_pref.unsqueeze(0)).abs()
#     same_mask = (pseudo_labels.unsqueeze(1) == pseudo_labels.unsqueeze(0)).to(dtype)
#     same_mask *= (1 - torch.eye(B, dtype=dtype, device=device))
#     repel_mask = (bias_diff > bias_eps).to(dtype)
#     pull_loss = (conf_weight * same_mask * dist.pow(2)).sum() / same_mask.sum().clamp(min=1.0)
#     margin = bias_diff * alpha
#     repel_term = F.relu(margin - dist) ** 2
#     repel_loss = (conf_weight * repel_mask * repel_term).sum() / repel_mask.sum().clamp(min=1.0)
#     total_loss = pull_loss + alpha * repel_loss
#     return total_loss


# def graph_regularization_loss(
#     features, pseudo_labels, class_preference, confidence,
#     alpha=1.0, tau=0.5, bias_eps=0.1, margin_base=1.0, adaptive=True
# ):
#     B = features.size(0)
#     dtype, device = features.dtype, features.device

#     # --- Normalize features ---
#     feat = F.normalize(features, dim=1)
#     dist = torch.cdist(feat, feat, p=2)  # Euclidean distance

#     # --- Compute bias & confidence weights ---
#     sample_pref = class_preference[pseudo_labels]                # [B]
#     bias_diff = (sample_pref.unsqueeze(1) - sample_pref.unsqueeze(0)).abs()  # [B,B]

#     same_mask = (pseudo_labels.unsqueeze(1) == pseudo_labels.unsqueeze(0)).to(dtype)
#     same_mask *= (1 - torch.eye(B, dtype=dtype, device=device))  # remove diag
#     repel_mask = (bias_diff > bias_eps).to(dtype)

#     conf = confidence.unsqueeze(1)
#     conf_weight = 0.5 * (conf + conf.T)  # [B,B]
#     # --- Split into pull/repel parts ---
#     # high-confidence encourages pull, low-confidence enhances repel
#     conf_pull = conf_weight
#     conf_repel = (1 - conf_weight)

#     # --- Pull loss (soft) ---
#     pull_term = conf_pull * same_mask * dist.pow(2)
#     pull_loss = pull_term.sum() / (same_mask.sum().clamp(min=1.0))

#     # --- Adaptive margin based on bias and distance statistics ---
#     if adaptive:
#         # margin shrinks as mean distance grows (stabilization)
#         mean_dist = dist.mean().detach()
#         margin = margin_base * (bias_diff + 0.2) / (1 + mean_dist)
#     else:
#         margin = margin_base * bias_diff

#     repel_term = F.relu(margin - dist) ** 2
#     repel_loss = (conf_repel * repel_mask * repel_term).sum() / repel_mask.sum().clamp(min=1.0)

#     # --- Combine ---
#     total_loss = pull_loss + alpha * repel_loss
#     return total_loss

def graph_regularization_loss(
    features,             # [B, D]
    pseudo_labels,        # [B]
    class_preference,     # [C]
    confidence,           # [B]
    alpha=0.5,           # repel loss weight
    margin=1.0,           # margin for repel
    bias_eps=0.1,         # preference difference threshold
    method='mean',        # confidence fusion: 'mean' | 'product' | 'exp'
    conf_thresh=0.7       # threshold for high-confidence gating
):
    B = features.size(0)
    # --- Cosine distance ---
    # feat = F.normalize(features, dim=1)
    # cos_sim = feat @ feat.T                      # [B, B], cosine similarity
    # dist = 1.0 - cos_sim                         # [B, B], cosine distance ∈ [0, 2]

    feat = F.normalize(features, dim=1)
    dist = torch.cdist(feat, feat, p=2).pow(2)  # [B, B]

    # --- Confidence weight ---
    conf = confidence.unsqueeze(1)  # [B, 1]
    if method == 'product':
        conf_weight = torch.sqrt(conf @ conf.T)
    elif method == 'mean':
        conf_weight = 0.5 * (conf + conf.T)
    elif method == 'exp':
        conf_weight = torch.exp((conf @ conf.T) - 1)
    else:
        raise ValueError("Unsupported confidence fusion strategy")

    # --- High-confidence gating mask ---
    mask = (confidence > conf_thresh).float().unsqueeze(1)  # [B, 1]
    gating = mask @ mask.T  # [B, B]
    conf_weight = conf_weight * gating

    # --- Class preference difference (normalized) ---
    sample_pref = class_preference[pseudo_labels]  # [B]
    bias_diff = torch.abs(sample_pref.unsqueeze(1) - sample_pref.unsqueeze(0))  # [B, B]
    bias_diff = bias_diff / (bias_diff.max() + 1e-6)  # normalize to [0, 1]
    repel_mask = (bias_diff > bias_eps).float()

    # --- Pull: same-class pairs ---

    
    bias_weight = torch.exp(1-sample_pref)
   
    same_class_mask = (pseudo_labels.unsqueeze(1) == pseudo_labels.unsqueeze(0)).float()
    same_class_mask = same_class_mask * (1 - torch.eye(B, device=feat.device))  # remove self-pairs
    pull_weight = same_class_mask * conf_weight * bias_weight
    pull_loss = (pull_weight * dist).sum() / same_class_mask.sum().clamp(min=1.0)
    
    repel_term = F.relu(margin - dist)
    repel_loss = (bias_diff * conf_weight * repel_term * repel_mask).sum() / repel_mask.sum().clamp(min=1.0)
    total_loss = pull_loss + alpha * repel_loss
    return total_loss


def load_clip_to_cpu(cfg):
    backbone_name = cfg.backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    # print(model)
    return model


def forward_reg_loss(pred_logits):
    pred_softmax = F.softmax(pred_logits, dim=1)
    ent_loss = - (pred_softmax * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()
    prob_mean = pred_softmax.mean(dim=0)
    ne_loss = (prob_mean * prob_mean.log()).sum()
    return ent_loss + ne_loss


def compute_adjustment_by_py(py, tro, device):
    adjustments = torch.log(py ** tro + 1e-12)
    adjustments = adjustments.to(device)
    return adjustments


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def label_smoothing(logits, labels, features, class_templates, sigma=0.1):
    batch_size, N = logits.shape
    tau = 0.1
    features = F.normalize(features, dim=1)
    class_templates = F.normalize(class_templates, dim=1)
    cos_sim = torch.matmul(features, class_templates.T) / 0.5  # [batch_size, N]
    # cos_sim = torch.matmul(features, class_templates.T)  # [batch_size, N]
    labels_one_hot = F.one_hot(labels, num_classes=N).float().to(logits.device)  # [batch_size, N]
    masked_cos_sim = cos_sim.masked_fill(labels_one_hot.bool(), float('-inf'))
    softmax_sim = F.softmax(masked_cos_sim, dim=1)
    smoothed_labels = sigma * softmax_sim
    smoothed_labels.scatter_(1, labels.unsqueeze(1), 1 - sigma)
    # print(torch.max(smoothed_labels, dim=1))
    # print(torch.sum(smoothed_labels, dim=1))
    # print(smoothed_labels)
    return smoothed_labels


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def text_embedding(filename, device):
    all_text_features = torch.load(filename)

    all_means = []
    all_covs = []
    for c in range(all_text_features.size(0)):
        cls_text_features = all_text_features[c].cpu().numpy()
        mean = np.mean(cls_text_features, axis=0)  # 1024
        cov = np.cov(cls_text_features.T)  # 1024 x 1024
        mean = torch.from_numpy(mean)
        cov = torch.from_numpy(cov)
        cov = cov.diag()
        all_means.append(mean)
        all_covs.append(cov)
    means = torch.stack(tuple(all_means), dim=0).half().to(device)
    covs = torch.stack(tuple(all_covs), dim=0).half().to(device)

    return means, covs


def proto(means, query):
    query = F.normalize(query, dim=1)
    T = 0.07
    means = F.normalize(means, dim=1)
    query_mean = query.mm(means.permute(1, 0))  # N*K
    logit = query_mean / T
    # logit = torch.softmax(logit, dim=1)
    return logit

def top5_accuracy_with_mask(gt_labels, predicted_probs, mask):
    # Get the indices of the top-5 predicted probabilities for each sample
    top5_preds = torch.topk(predicted_probs, k=20, dim=1, largest=False).indices
    # Check if the ground truth labels are within the top-5 predictions
    matches = torch.any(top5_preds == gt_labels.unsqueeze(1), dim=1)
    # Apply the mask to ignore certain samples
    masked_matches = matches * mask
    # Calculate the accuracy for non-masked samples
    if masked_matches.numel() == 0:
        return 0.0  # Handle edge case: all samples are masked
    # top5_accuracy = torch.sum(masked_matches.float())
    return masked_matches



def entropy_confidence(probs, eps=1e-4):
    device = probs.device
    dtype = probs.dtype
    C = probs.size(1)
    probs = probs.clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=-1)
    logC = torch.log(torch.tensor(C, device=device, dtype=dtype))
    confidence = 1.0 - entropy / logC
    confidence = confidence.clamp(0.0, 1.0)
    return confidence



class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.num_epochs = cfg.num_epochs
        self.output_dir = cfg.output_dir

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        # self.evaluator = Evaluator(cfg, self.cls_num_list)
        self.best_result = -np.inf
        self._writer = None
        self.th = cfg.th
        
        class_list = []
        for i in range(cfg.DATA.NUMBER_CLASSES):
            class_list.append(str(i))

        title = 'PEL-SSL-' + cfg.DATA.NAME
        self.logger = Logger(os.path.join(cfg.output_dir, cfg.text_name), title=title)
        self.logger.set_names(['Top1 acc', 'Best Top1 acc', 'epoch'])

    def build_data_loader(self):
        cfg = self.cfg
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[cfg.DATA.NAME](
            cfg)

        self.num_classes = cfg.DATA.NUMBER_CLASSES
        self.classnames = labeled_dataset.classes

        # self.sampled_cls_num_list = self.cls_num_list
        self.train_label_loader = DataLoader(labeled_dataset, num_workers=cfg.DATA.NUM_WORKERS,
                                             batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, drop_last=True,
                                             pin_memory=True)

        self.train_unlabel_loader = DataLoader(unlabeled_dataset, num_workers=cfg.DATA.NUM_WORKERS,
                                               batch_size=cfg.DATA.BATCH_SIZE * self.cfg.DATA.MU_U, shuffle=True,
                                               drop_last=True, pin_memory=True)

        self.test_loader = DataLoader(
            test_dataset,
            num_workers=cfg.DATA.NUM_WORKERS,
            sampler=SequentialSampler(test_dataset),
            batch_size=100)

    def build_model(self):
        cfg = self.cfg
        # classnames = self.classnames

        print(f"Loading CLIP (backbone: {cfg.backbone})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        print(cfg.prec)

        assert cfg.prec in ["fp16", "fp32", "amp"]
        if cfg.prec == "fp32" or cfg.prec == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        if cfg.template is not None:
            temp = cfg.template
        else:
            temp = "a photo of a {}."
        print(temp)
        print(self.classnames)
        prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
        # prompts = [c for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        # data_root = '/mnt/raid1/zhengna/CVPR_2026/datasets'
        # # text_data = 
        # text_data_path = os.path.join(data_root, self.cfg.dataset+'.json')
        # text_data = json.load(open(text_data_path,'r'))
        # text_data_list = []
        # keys = text_data.keys()
        
        

        # prompts = []
        # for c in self.classnames:
        #     name = c.replace("_"," ")
        #     if name in text_data:
        #         texts = text_data[name]
        #         prompts.extend(texts) 
        #     else:
        #         print(f"{name} not found")
        # prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts])
        # prompts = prompts.to(self.device)
        
        # print(text_data_list)
        # text_data_list = torch.cat()
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        num_classes = len(self.classnames)
        num_prompts = text_features.shape[0] // num_classes
        text_features = text_features.view(num_classes, num_prompts, -1)
        text_features = text_features.mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features
        print(self.text_features.shape)
        print("Building model")
        self.model = Model(cfg, clip_model, self.text_features)
        self.tuner = self.model.tuner
        self.clip_model = clip_model
        self.dtype = clip_model.dtype

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total params: {total_params}')
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        print(f'Tuned params: {tuned_params}')
        head_params = sum(p.numel() for p in self.tuner.head.parameters())
        tuned_params_without_head = tuned_params - head_params
        print(f'Tuned params (w/o head): {tuned_params_without_head}')

        self.optim = torch.optim.SGD(self.tuner.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                     momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, float(cfg.num_epochs))
        self.scaler = GradScaler() if cfg.prec == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(cfg.gpu)
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)



    def train(self):
        global best_acc
        global best_zs_acc
        global best_acc1
        # added for threshold filter
        filtered = torch.tensor([]).to(device=self.device)
        # added for threshold filter
        filtered_batch = torch.tensor([]).to(self.device)
        pseudo_label_acc = torch.tensor([]).to(self.device)
        unselected_1 = torch.tensor([]).to(self.device)
        unselected_2 = torch.tensor([]).to(self.device)

        if self.cfg.resume:
            directory = self.cfg.resume
            self.start_epoch = self.resume_model_if_exist(directory)

        self.tuner.head1.weight.data = self.tuner.head.weight.data.clone()
        # Initialize summary writer
        writer_dir = os.path.join(self.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        self.w_con = self.cfg.w_con
        ulab_len = len(self.train_unlabel_loader.dataset)

        self.alpha = self.cfg.alpha
        self.betabase = torch.ones((self.num_classes, self.num_classes)).to(self.device)
        self.betabase[torch.arange(self.num_classes), torch.arange(self.num_classes)] = 0.0

        self.smoothing = self.cfg.smoothing
        self.th_min = self.cfg.th_min

        self.time_start = time.time()

        names = {'clip': '_clipL.pt', 'bert': '_BertL.pt', 'mT5': '_mT5L.pt'}
        filename = self.cfg.dataset + names[self.cfg.text_model_name]
        print('using text model: ', filename)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            
        total_steps = self.cfg.total_steps * self.num_epochs
        self.warmup_iters = int(0.25 * total_steps)
        self.model.compute_static_bias(self.train_unlabel_loader, num_batches=250)
        for self.epoch in range(self.start_epoch, self.num_epochs):
            
            self.tuner.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            # added for threshold filter
            filtered_epoch = torch.tensor([]).to(device=self.device)
            epoch_acc = torch.tensor([]).to(self.device)
            unselect_acc_1 = torch.tensor([]).to(self.device)
            unselect_acc_2 = torch.tensor([]).to(self.device)
            filter = torch.tensor([]).to(self.device)
            self.num_batches = self.cfg.total_steps
            label_loader_iter = cycle(self.train_label_loader)
            unlabel_loader_iter = cycle(self.train_unlabel_loader)

            selected_label = torch.ones((ulab_len,), dtype=torch.long, ) * -1
            selected_label = selected_label.to(self.device)
            classwise_acc = torch.zeros((self.num_classes,)).to(self.device)
            end = time.time()

            for self.batch_idx in range(self.cfg.total_steps):
                
                data_time.update(time.time() - end)

                (inputs_x, targets_x, _) = next(label_loader_iter)
                batch_size = inputs_x.shape[0]
                ((inputs_u_w, inputs_u_s, inputs_u_s1), u_real, uidx) = next(unlabel_loader_iter)

                uidx = uidx.to(self.device)
                targets_x = targets_x.to(self.device)
                targets_x = targets_x.to(torch.long)
                pseudo_counter = Counter(selected_label.tolist())

                inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1 = inputs_x.to(self.device), inputs_u_w.to(self.device), \
                    inputs_u_s.to(self.device), inputs_u_s1.to(self.device)

                inputs = interleave(
                    torch.cat((inputs_x, inputs_u_w, inputs_u_s, inputs_u_s1)), 3 * self.cfg.DATA.MU_U + 1)
                feat = self.model(inputs)
                feat = de_interleave(feat, 3 * self.cfg.DATA.MU_U + 1)
                feat_l_w = feat[:batch_size]
                feat_u_w, feat_u_s, feat_u_s1 = feat[batch_size:].chunk(3)
                output = self.tuner.head(feat)
                output_x = output[:batch_size]
                output_u_w, output_u_s, output_u_s1 = output[batch_size:].chunk(3)
                # output1 = self.tuner.head1(feat.detach())
                # output1_x = output1[:batch_size]
                # output1_u_w, output1_u_s, output1_u_s1 = output1[batch_size:].chunk(3)
                del feat
                del output
                # del output1

                pseu = torch.softmax(output_u_w.detach(), dim=-1)
                conf, targets_u = torch.max(pseu, dim=-1)
                mask = conf.ge(self.th)
                mask_twice = torch.cat([mask, mask], dim=0).to(self.device)
                mask_twice = mask_twice.half()
                output_u_s_twice = torch.cat([output_u_s, output_u_s1], dim=0).to(self.device)
                targets_u_twice = torch.cat([targets_u, targets_u], dim=0).to(self.device)
                L_reg = forward_reg_loss(output_u_s_twice) + forward_reg_loss(output_u_w)
                
                
                bias, logit_adjust = self.model.masking(pseu, softmax_x_ulb=False, factor=True)
                bias_pred = bias[targets_u] 

                # alpha_eff = (1 - conf)**2
                # ent_conf = entropy_confidence(pseu)
                # confidence = (ent_conf ** alpha_eff) * (conf ** (1 - alpha_eff))
                self.cur_bias = bias
                alpha = self.cfg.alpha
                sigma = self.cfg.sigma
                lamda = self.cfg.lamda
                tem = self.cfg.tem
                # print(alpha,'--------------------')
                ent_conf = entropy_confidence(pseu)
                confidence = (ent_conf ** alpha) * (conf ** (1 - alpha))
                # confidence  = conf# * (ent_conf ** alpha)
                bias_pred_norm = bias_pred / (bias_pred.mean() + 1e-6)

                # scores = confidence * ((torch.exp((1-bias_pred) * tem))**(0.1))
                scores = confidence * ((torch.exp((1-bias_pred) * tem)))
                scores_twice = torch.cat([scores,scores],dim=0)
                Lx = F.cross_entropy(output_x, targets_x, reduction='mean')
                self.model.update_ema_encoder()
                feats = self.model.enqueue_from_ema(inputs_u_w, output_u_w, feat_u_w)
                feats = F.normalize(feats, dim=1)
                d_j = self.model.compute_relative_distance(feats.detach(), targets_u)
                r_bias = torch.exp((1 - bias_pred) * d_j.to(bias_pred.dtype))
                r_bias_twice = torch.cat([r_bias, r_bias], dim=0)
                # self.model.enqueue_from_logits(feat_u_w, output_u_w)
                mask_ent = confidence.ge(self.th)
                mask_ent_twice = torch.cat([mask_ent, mask_ent], dim=0).to(self.device)
                mask_ent_twice = mask_ent_twice.half()
                

                if torch.sum(mask_twice) > 0:
                    # Lu = (F.cross_entropy(output_u_s_twice+logit_adjust, targets_u_twice,
                    #                       reduction='none')* scores_twice * mask_twice).mean()
                    Lu = (F.cross_entropy(output_u_s_twice+logit_adjust, targets_u_twice,
                                          reduction='none') * scores_twice* mask_twice).mean()
                    # Lu = (F.cross_entropy(output_u_s_twice+logit_adjust, targets_u_twice,
                    #                       reduction='none')  * mask_twice).mean()
                    # Lu = (F.cross_entropy(output_u_s_twice+logit_adjust, targets_u_twice,
                    #                       reduction='none') * scores_twice * mask_twice).mean()
                    # graph_loss = graph_regularization_loss(features=feat_u_w, pseudo_labels=targets_u, class_preference=bias, confidence=confidence.detach()) #+ \
                
                    queue, queue_labels, queue_logits = self.model.get_queue()
                    queue_prob = torch.softmax(queue_logits, dim=1)  # [M, C]
                    queue_conf, _ = queue_prob.max(dim=1) 
                    entropy_conf = entropy_confidence(queue_prob)
                    queue_confidence = (entropy_conf ** alpha) * (queue_conf ** (1 - alpha))
                    graph_loss = graph_regularization_loss_memory_2(features=feat_u_w,pseudo_labels=targets_u,class_preference=bias,confidence=confidence.detach(), memory_feats=queue,memory_labels=queue_labels, memory_conf=queue_conf.detach(), alpha = sigma)
                    
                
                else:
                    Lu = 0
                    graph_loss = 0
                
                loss = Lx + Lu * self.w_con + graph_loss * lamda * self.model.time_p+  0.01 * L_reg #+ graph_loss * lamda * self.model.time_p
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                epoch_acc = torch.cat((epoch_acc, (targets_u == u_real.to(self.device)).float()), dim=0)
               
                filtered_batch = torch.cat((filtered_batch, mask_twice.float().mean(-1, keepdim=True)), dim=0)
                filtered_epoch = torch.cat((filtered_epoch, mask_twice.float()), dim=0)
                
                current_logit_scale = self.tuner.head.logit_scale.item()
                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())

                batch_time.update(time.time() - end)

                meet_freq = (self.batch_idx + 1) % self.cfg.print_freq == 0
                only_few_batches = self.num_batches < self.cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += self.num_batches - self.batch_idx - 1
                    nb_remain += (
                                         self.num_epochs - self.epoch - 1
                                 ) * self.num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{self.epoch + 1}/{self.num_epochs}]"]
                    info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"s {current_logit_scale:.4f}"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = self.epoch * self.num_batches + self.batch_idx
                self._writer.add_scalar("train/loss", loss_meter.avg, n_iter)
                self._writer.add_scalar("train/acc", acc_meter.avg, n_iter)
                self._writer.add_scalar("train/lr", current_lr, n_iter)

                if (self.batch_idx + 1) == self.num_batches:
                    self.sched.step()

                end = time.time()

            last_epoch = (self.epoch + 1) == self.num_epochs
            meet_checkpoint_freq = (
                (self.epoch + 1) % self.cfg.checkpoint_freq == 0
                if self.cfg.checkpoint_freq > 0 else False
            )

            if meet_checkpoint_freq or last_epoch:
                self.save_model(self.epoch, self.output_dir)

            # added for threshold filter
            print(f"{filtered_epoch.min()=} {filtered_epoch.max()=} {filtered_epoch.mean()=} {filtered_epoch.std()=} {filtered_epoch.median()=}")
            filtered = torch.cat((filtered, filtered_epoch.mean(-1, keepdim=True)), dim=-1)
            pseudo_label_acc = torch.cat((pseudo_label_acc, epoch_acc.mean(-1, keepdim=True)), dim=-1)
            print(pseudo_label_acc)

            unselect_acc_1 = torch.sum(unselect_acc_1) / torch.sum(filter)
            unselect_acc_2 = torch.sum(unselect_acc_2) / torch.sum(filter)
            unselect_acc_1 = unselect_acc_1.unsqueeze(0)
            unselect_acc_2 = unselect_acc_2.unsqueeze(0)
            unselected_1 = torch.cat([unselected_1, unselect_acc_1], dim=-1)
            unselected_2 = torch.cat([unselected_2, unselect_acc_2], dim=-1)

            acc_now = self.test()
            best_acc = max(best_acc, acc_now)
            self.logger.append([acc_now, best_acc, self.epoch + 1])
            # if self.epoch % 5 == 0 and self.epoch != 0:
            #     self.save_test_snapshot(self.epoch)
            # elif last_epoch:
            #     self.save_test_snapshot(self.epoch)


        
        print("Finish training")
        # added for threshold filter
        # assert filtered.shape[0] == self.cfg.num_epochs
        torch.save(dict(epoch=filtered.cpu(),
                        batch=filtered_batch.cpu(),
                        pseudo_acc=pseudo_label_acc.cpu()),
                   self.cfg.output_dir + '/pseudo_acc.pt')


        print("Deploy the last-epoch model for testing")
        # self.save_epoch_snapshot(
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self._writer.close()

        self.logger.close()

    @torch.no_grad()
    def test(self):
        self.tuner.eval()
        print(f"Evaluate on the test set")

        preds = np.array([])
        targets = np.array([])
        feats = []
        labels = torch.tensor([], dtype=torch.long)
        outputs = []

        for batch in tqdm(self.test_loader, ascii=True):
            image = batch[0].to(self.device, non_blocking=True)
            label = batch[1].to(self.device, non_blocking=True)

            feat = self.model(image)
            output = self.tuner.head(feat)

            feats.append(feat.cpu())

            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            preds = np.append(preds, pred.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())

            outputs.append(output.cpu())
            labels = torch.cat([labels, label.cpu()], dim=-1)

        targets = targets.astype(int)
        preds = preds.astype(int)

        outputs = torch.cat(outputs, dim=0)
        acc = sum(targets == preds) / len(targets)

        # ── per-class accuracy ────────────────────────────────────────
        targets_t = torch.tensor(targets)
        preds_t   = torch.tensor(preds)
        correct   = torch.zeros(self.model.class_num)
        total     = torch.zeros(self.model.class_num)
        for c in range(self.model.class_num):
            mask       = (targets_t == c)
            correct[c] = (preds_t[mask] == c).sum()
            total[c]   = mask.sum()
        per_class_acc = correct / (total + 1e-6)

        # 打出 scatter 和 label_hist 对应类的 accuracy
        scatter_norm  = self.model.compute_scatter()
        hist          = self.model.label_hist

        scatter_top5  = scatter_norm.cpu().topk(5).indices
        scatter_bot5  = scatter_norm.cpu().topk(5, largest=False).indices
        hist_top5     = hist.cpu().topk(5).indices

        # print(f"scatter top5 类: {scatter_top5.tolist()} → acc: {per_class_acc[scatter_top5].tolist()}")
        # print(f"scatter bot5 类: {scatter_bot5.tolist()} → acc: {per_class_acc[scatter_bot5].tolist()}")
        # print(f"label_hist top5 类: {hist_top5.tolist()} → acc: {per_class_acc[hist_top5].tolist()}")
        # print(f"全局平均 accuracy: {per_class_acc.mean():.4f}")
        # ─────────────────────────────────────────────────────────────

        feats_tensor = torch.cat(feats, dim=0)
        return acc
    # def test(self):
    #     self.tuner.eval()
    #     print(f"Evaluate on the test set")

    #     preds = np.array([])
    #     targets = np.array([])
    #     feats = []
    #     labels = torch.tensor([], dtype=torch.long)
    #     outputs = []

    #     for batch in tqdm(self.test_loader, ascii=True):
    #         image = batch[0].to(self.device, non_blocking=True)
    #         label = batch[1].to(self.device, non_blocking=True)

    #         feat = self.model(image)
    #         output = self.tuner.head(feat)

    #         feats.append(feat.cpu())

    #         prob = F.softmax(output, dim=1)
    #         conf, pred = prob.max(1)
    #         preds = np.append(preds, pred.cpu().numpy())
    #         targets = np.append(targets, label.cpu().numpy())

    #         outputs.append(output.cpu())
    #         labels = torch.cat([labels, label.cpu()], dim=-1)

    #     targets = targets.astype(int)
    #     preds = preds.astype(int)

    #     outputs = torch.cat(outputs, dim=0)
    #     acc = sum(targets == preds) / len(targets)
    #     print(outputs.shape)

    #     feats_tensor = torch.cat(feats, dim=0)
        

    #     # print(feats_tensor.shape)
    #     return acc
    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        tuner_dict = self.tuner.state_dict()
        optim_dict = self.optim.state_dict()
        sched_dict = self.sched.state_dict()
        save_checkpoint(
            {
                "state_dict": tuner_dict,
                "epoch": epoch + 1,
                "optimizer": optim_dict,
                "scheduler": sched_dict,
                "val_result": val_result
            },
            os.path.join(directory + f'/epoch_{epoch + 1}', "tuner"),
            is_best=is_best,
            model_name=model_name,
        )

    def resume_model_if_exist(self, directory):
        file_missing = False

        path = os.path.join(directory, "tuner")
        if not os.path.exists(path):
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        path = os.path.join(directory, "tuner")
        start_epoch = resume_from_checkpoint(
            path, self.tuner, self.optim, self.sched
        )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        model_path = os.path.join(directory, "tuner", model_file)

        if not os.path.exists(model_path):
            raise FileNotFoundError('Model not found at "{}"'.format(model_path))

        checkpoint = load_checkpoint(model_path, self.device)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]

        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        print("Loading weights to {} " 'from "{}" (epoch = {})'.format("tuner", model_path, epoch))
        # set strict=False
        self.tuner.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def get_comparison_data(self):
        # operate on which data? Unlabelled weak data.
        self.tuner.eval()
        print(f"Getting data for comparison plots")
        preds = torch.tensor([])
        targets = torch.tensor([])
        confidences = torch.tensor([])
        rect_confidences = torch.tensor([])
        # probs = list()
        prob = torch.tensor([])
        outputs = []
        feats = []
        for batch in tqdm(self.train_unlabel_loader, ascii=True):
            ((images, _, _), label, _) = batch
            # images, label, _ = batch
            # print(f"{images.shape=}")

            images = images.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            feat = self.model(images)
            output = self.tuner.head(feat)
            # del feat

            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            confidences = torch.cat((confidences, conf.cpu()), dim=-1)
            preds = torch.cat((preds, pred.cpu()), dim=-1)
            targets = torch.cat((targets, label.cpu()), dim=-1)
            outputs.append(output.cpu())
            feats.append(feat.cpu())
            res = (pred == label).cpu()
            acc = (res).float().mean()
            k = acc / conf.mean().cpu()
            rect_conf = k * conf.cpu()
            rect_confidences = torch.cat((rect_confidences, rect_conf), dim=-1)

        feats = torch.cat(feats, dim=0)
        outputs = torch.cat(outputs, dim=0)
        # plot 1
        class_indices, counts = torch.unique(preds, return_counts=True)
        sorted_counts, count_indices = torch.sort(counts, descending=True)
        class_indices = class_indices[count_indices]

        # plot 2
        average_confidence = confidences.mean()

        # plot 3 - ece
        ece_scorer = _ECELoss()
        ece = ece_scorer(preds=preds, targets=targets, confs=rect_confidences)

        # plot 4
        true_rc = rect_confidences[targets == preds]
        false_rc = rect_confidences[~(targets == preds)]


    @torch.no_grad()
    def save_test_snapshot(self, epoch, use_half=True):
        """
        Save a snapshot of the test set (features, logits, labels, confidence, entropy).
        Used mainly for visualization or analysis on the test distribution.
        """

        # ---- Sanity check ----
        assert hasattr(self, "test_loader"), "❌ self.test_loader not found. Please define self.test_loader first."
        dataloader = self.test_loader

        self.model.eval()
        self.tuner.eval()

        all_feats, all_logits = [], []
        all_labels, all_pseudo = [], []
        all_conf, all_entropy = [], []

        for batch in tqdm(dataloader, desc=f"[Test] Saving snapshot (epoch {epoch})"):
            # Unpack batch
            images = batch[0].to(self.device, non_blocking=True)
            labels = batch[1].to(self.device, non_blocking=True)

            # ---- Forward ----
            feats = self.model(images)                  # [B, D]
            logits = self.tuner.head(feats)             # [B, C]
            probs = F.softmax(logits, dim=1)
            conf, pseudo = probs.max(dim=1)
            ent_conf = entropy_confidence(probs)

            # ---- Collect ----
            all_feats.append(feats.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_pseudo.append(pseudo.cpu())
            all_conf.append(conf.cpu())
            all_entropy.append(ent_conf.cpu())

        # ---- Concatenate ----
        feats = torch.cat(all_feats)
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        pseudo = torch.cat(all_pseudo)
        confs = torch.cat(all_conf)
        entropy_conf = torch.cat(all_entropy)

        # ---- Save ----
        save_dict = {
            'epoch': epoch,
            'features': feats.half() if use_half else feats,
            'logits': logits.half() if use_half else logits,
            'true_labels': labels,
            'pseudo_labels': pseudo,
            'confidence': confs,
            'entropy_conf': entropy_conf
        }

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, f"epoch_{epoch:03d}_snapshot_test_our_2.pt")
        torch.save(save_dict, save_path)
        print(f" Saved test snapshot → {save_path}")

    @torch.no_grad()
    def save_epoch_snapshot(self, epoch, dataloader=None):
       
        if dataloader is None:
            dataloader = self.train_unlabel_loader

        all_feats, all_logits, all_labels, all_pseudo, all_conf, all_entropy = [], [], [], [], [], []

        self.model.eval()
        self.tuner.eval()

        for batch in tqdm(dataloader, desc=f"[Epoch {epoch}] Saving snapshot"):
            ((images, _, _), labels, _) = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 提取特征与 logits
            feats = self.model(images)                   # [B, D]
            logits = self.tuner.head(feats)              # [B, C]
            probs = F.softmax(logits, dim=1)
            conf, pseudo = probs.max(dim=1)
            ent_conf = entropy_confidence(probs)

            all_feats.append(feats.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_pseudo.append(pseudo.cpu())
            all_conf.append(conf.cpu())
            all_entropy.append(ent_conf.cpu())

        
        feats = torch.cat(all_feats)
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        pseudo = torch.cat(all_pseudo)
        confs = torch.cat(all_conf)
        entropy_conf = torch.cat(all_entropy)

        # bias, _ = self.model.masking(F.softmax(logits, dim=1), softmax_x_ulb=False, factor=True)
        bias = self.cur_bias
        bias = bias.cpu()
        class_counts = torch.bincount(pseudo, minlength=self.num_classes).float()
        class_freq = class_counts / class_counts.sum()

        save_dict = {
            'epoch': epoch,
            'features': feats.half(),
            'logits': logits.half(),
            'true_labels': labels,
            'pseudo_labels': pseudo,
            'bias': bias,
            'class_freq': class_freq,
            'confidence': confs,
            'entropy_conf': entropy_conf
        }

        save_path = os.path.join(self.output_dir, f"epoch_{epoch}_snapshot_our_2.pt")
        torch.save(save_dict, save_path)
        print(f" Saved epoch snapshot → {save_path}")



    @torch.no_grad()
    def get_feature(self):
        # operate on which data? Unlabelled weak data.
        preds = torch.tensor([])  
        targets = torch.tensor([])  
        confidences = torch.tensor([])  
        rect_confidences = torch.tensor([])  
        prob = torch.tensor([])  
        outputs = []  
        feats = []  # 用来存储特征

        for batch in tqdm(self.train_unlabel_loader, ascii=True):
            ((images, _, _), label, _) = batch
            images = images.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            feat = self.model(images)  # 获取当前批次的特征
            outputs.append(feat.cpu())
            targets = torch.cat((targets, label.cpu()), dim=-1)
            feats.append(feat.cpu())  # 将特征存储到列表中

        feats = torch.cat(feats, dim=0)  # 将所有批次的特征连接起来

        # 保存特征和标签
        torch.save({'features': feats, 'labels': targets}, 'features_and_labels_epoch_{}.pt'.format(self.epoch))
        print(f"Saved features and labels for epoch {self.epoch}")

