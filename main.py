import os
import random
import argparse
import numpy as np
import torch
import setproctitle
from utils.config import _C as cfg
from utils.logger import setup_logger

from trainer import Trainer
# from negative import Trainer
# from naive import 
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)                       # Python 自带的随机模块
    np.random.seed(seed)                    # numpy 随机数
    torch.manual_seed(seed)                 # CPU 上的 torch
    torch.cuda.manual_seed(seed)            # GPU 上的 torch
    torch.cuda.manual_seed_all(seed)        # 多 GPU 情况下的 seed

    torch.backends.cudnn.deterministic = True   # 让 cudnn 使用确定性算法
    torch.backends.cudnn.benchmark = False      # 禁用自动调参（否则每次会变）

# 在训练前调用


def main(args):

    # set_seed(42)
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    seed_random = torch.initial_seed()
    # print("Seed:", torch.initial_seed())
    if cfg.seed is not None:
        seed = cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    imbl = cfg.DATA.IMB_L
    imbu = cfg.DATA.IMB_U
    numl = cfg.DATA.NUM_L
    if cfg.output_dir is None:
        cfg.output_dir = os.path.join("./output", os.path.basename(args.cfg).rstrip(".yaml"))
    else:
        cfg.output_dir = os.path.join(cfg.output_dir, cfg.DATA.NAME, f"NUML{numl}_imbl{imbl}_imbu{imbu}")

    print("** Config **")
    print(cfg)
    # clean_title = f"python main.py  --alpha {cfg.alpha} --bank_size {cfg.bank_size}"
    # setproctitle.setproctitle(clean_title)
    if cfg.eval_only:
        cfg.model_dir = cfg.model_dir if cfg.model_dir is not None else cfg.output_dir
        cfg.load_epoch = cfg.load_epoch if cfg.load_epoch is not None else cfg.num_epochs
        trainer.load_model(cfg.model_dir, epoch=cfg.load_epoch)
        trainer.test()
        return

    setup_logger(cfg.output_dir)
    trainer = Trainer(cfg)
    print(seed_random)
    print("start training...------------------------------")
    trainer.train()
    print(seed_random)
    # trainer.get_comparison_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="", help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
