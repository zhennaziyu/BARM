import os
import math
import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# from data.tranforms import TransformTrainCifar, ResizeImage
# from data.cifar100 import x_u_split
# from .data import TransformFixMatch_ws, train_split_l

from .randaugment import RandAugmentMC

# if __name__ == "__main__":
#     print("test")
class TransformFixMatch_ws(object):
    def __init__(self, mean, std, img_size=32):
        resize_dim = 256
        crop_dim = 224
        self.weak = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim * 0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip()])

        self.strong = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim*0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong1)

def train_split_l(labels, n_labeled_per_class, cfg):
    labels = np.array(labels)
    train_labeled_idxs = []
    # train_unlabeled_idxs = []
    for i in range(cfg.DATA.NUMBER_CLASSES):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        # train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs

def x_u_split(args, labels):
    # print(labels)
    # print('////////')

    label_per_class = args.DATA.NUM_L

    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    # print(len(labels))
    # print('-----------------')
    for i in range(args.DATA.NUMBER_CLASSES):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.DATA.NUM_L*args.DATA.NUMBER_CLASSES
    # print(args.DATA.NUM_L*args.DATA.NUMBER_CLASSES)
    if args.expand_label or args.DATA.NUM_L*args.DATA.NUMBER_CLASSES < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.test_interval / args.DATA.NUM_L*args.DATA.NUMBER_CLASSES)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

class Food101Dataset(Dataset):
    def __init__(self, df, root, label_list, transforms_list, 
                 return_idx=True, indices=None):
        super(Food101Dataset, self).__init__()
        self.dir = Path(root)
        self.data = df
        self.return_idx = return_idx
        self.label_list = label_list

        self.classes = label_list
        # self.indices = self.data.index.tolist()
        self.targets = self.data.labels.apply(
                    lambda x: self.label_list.index(x))
        self.transforms = transforms_list

        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # row = self.data.iloc[idx]
        # image = Image.open(self.dir/f"images/{row.path}.jpg").convert('RGB')
        # label = self.label_list.index(row.labels)
        #
        # if self.transforms is not None:
        #     lm = self.transforms(image)
        # image.close()
        row = self.data.iloc[idx]

        label = self.label_list.index(row.labels)
        with Image.open(self.dir / f"images/{row.path}.jpg") as img:
            im = img.convert('RGB')
            if self.transforms is not None:
                im = self.transforms(im)

        if self.return_idx:
            return im, label, idx

        return im, label


def get_food101_alt(args):
    root = Path(args.root)/'food-101'

    resize_dim = 256
    crop_dim = 224
    dataset_mean = (0.485, 0.456, 0.406)
    dataset_std = (0.229, 0.224, 0.225)

    labels = pd.read_csv(root/'meta/labels.txt', header=None)
    labels.columns = 'names'.split()
    labels = labels.assign(names=labels.names.apply(
                        lambda x: x.lower().replace(' ', '_')))
    labels = labels.names.tolist()
    
    train_df = pd.read_csv(root/"meta/train.txt", header=None)
    train_df.columns=['path']
    train_df = train_df.assign(
                labels=train_df.path.apply(lambda x: x.split('/')[0]))
    
    test_df = pd.read_csv(root/"meta/test.txt", header=None)
    test_df.columns=["path"]
    test_df = test_df.assign(
                labels=test_df.path.apply(lambda x: x.split('/')[0]))
    

    train_targets = train_df.labels.apply(lambda x: labels.index(x))
    labeled_ratio = args.DATA.LABEL_RATIO
    num_labeled_per_cls = [args.DATA.NUM_L for _ in range(args.DATA.NUMBER_CLASSES)]
    train_labeled_idxs = train_split_l(train_targets, num_labeled_per_cls, args)
    
    print("train_labeled_idxs: ", len(train_labeled_idxs))
    # print('---------------------------------')
    # print("train_unlabeled_idxs: ", len(train_unlabeled_idxs))

    # if args.data_return_index:
    #     return_index = True
    # else:
    #     return_index = False

    return_index=True

    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)])

    train_labelled_dataset = Food101Dataset(
                                df=train_df,
                                root=root,
                                label_list=labels,
                                transforms_list=transform_labeled,
                                indices=train_labeled_idxs,
                                return_idx=return_index)

    train_unlabelled_dataset = Food101Dataset(
                                df=train_df,
                                root=root,
                                label_list=labels,
                                transforms_list=TransformFixMatch_ws(
                                    mean=dataset_mean, std=dataset_std),
                                indices=None,
                                return_idx=return_index)
    
    test_dataset = Food101Dataset(
                                df=test_df,
                                root=root,
                                label_list=labels,
                                transforms_list=transform_val,
                                indices=None,
                                return_idx=True,                        
                            )
    return train_labelled_dataset, train_unlabelled_dataset, test_dataset