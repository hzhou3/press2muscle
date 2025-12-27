import os
from argparse import ArgumentParser
from yacs.config import CfgNode as CN


def get_args():
    parser = ArgumentParser(description='Train Network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="",
                        type=str)
    parser.add_argument('--debug',
                        action="store_true",
                        )
    parser.add_argument('--vis',
                        action="store_true",
                        )
    parser.add_argument('-e', '--experiment',
                        help='output folder name',
                        default="",
                        type=str)         
    args = parser.parse_args()
    print(f"using config {args.cfg}")
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.cfg)
    name = args.cfg.split('/')[1].split('.')[0]
    cfg.NAME = name
    cfg.EXPERIMENT = args.experiment

    if cfg.EXPERIMENT == "":
        cfg.SAVE_DIR = os.path.join("outputs", name+"_"+cfg.PROTOCOL)
    else:
        cfg.SAVE_DIR = os.path.join("outputs", cfg.EXPERIMENT)
    
    cfg.VIS = args.vis
    return cfg
