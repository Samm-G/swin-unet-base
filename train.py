import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config

import yaml

parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='.\\data\\project_TransUNet\\data\\Synapse', help='root dir for data')
# parser.add_argument('--dataset', type=str,
#                     default='Synapse', help='experiment_name')
# parser.add_argument('--list_dir', type=str,
#                     default='./lists/lists_Synapse', help='list dir')
# parser.add_argument('--num_classes', type=int,
#                     default=9, help='output channel of network')
# parser.add_argument('--output_dir', type=str, help='output dir')                   
# parser.add_argument('--max_iterations', type=int,
#                     default=30000, help='maximum epoch number to train')
# parser.add_argument('--max_epochs', type=int,
#                     default=150, help='maximum epoch number to train')
# parser.add_argument('--batch_size', type=int,
#                     default=24, help='batch_size per gpu')
# parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
# parser.add_argument('--deterministic', type=int,  default=1,
#                     help='whether use deterministic training')
# parser.add_argument('--base_lr', type=float,  default=0.01,
#                     help='segmentation network learning rate')
# parser.add_argument('--img_size', type=int,
#                     default=224, help='input patch size of network input')
# parser.add_argument('--seed', type=int,
#                     default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
# parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+',
#     )
# parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
# parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
#                     help='no: no cache, '
#                             'full: cache all data, '
#                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
# parser.add_argument('--resume', help='resume from checkpoint')
# parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
# parser.add_argument('--use-checkpoint', action='store_true',
#                     help="whether to use gradient checkpointing to save memory")
# parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
#                     help='mixed precision opt level, if O0, no amp is used')
# parser.add_argument('--tag', help='tag of experiment')
# parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
# parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)
del args

if config.MODEL.SWIN.QK_SCALE == 'None':
    config.MODEL.SWIN.QK_SCALE = None

if "Synapse" in list(config.DATASETS.keys()):
    config.DATASETS.Synapse.ROOT_PATH = os.path.join(config.DATASETS.Synapse.ROOT_PATH, "train_npz")

if __name__ == "__main__":
    if not config.TRAIN.DETERMINISTIC:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(config.TRAIN.SEED)
    np.random.seed(config.TRAIN.SEED)
    torch.manual_seed(config.TRAIN.SEED)
    torch.cuda.manual_seed(config.TRAIN.SEED)

    dataset_name = list(config.DATASETS.keys())[0]

    # dataset_config = {
    #     'Synapse': {
    #         'root_path': args.root_path,
    #         'list_dir': './lists/lists_Synapse',
    #         'num_classes': 9,
    #     },
    # }

    if config.TRAIN.BATCH_SIZE != 24 and config.TRAIN.BATCH_SIZE % 6 == 0:
        config.TRAIN.BASE_LR *= config.TRAIN.BATCH_SIZE / 24
    # args.num_classes = config.DATASETS.Synapse.NUM_CLASSES
    # args.root_path = config.DATASETS.Synapse.ROOT_PATH
    # args.list_dir = config.DATASETS.Synapse.LIST_DIR

    if not os.path.exists(config.TRAIN.OUTPUT_DIR):
        os.makedirs(config.TRAIN.OUTPUT_DIR)
    net = ViT_seg(config, img_size=config.DATA.IMG_SIZE, num_classes=config.DATASETS.Synapse.NUM_CLASSES).cuda()
    # net.load_from(config)

    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](config, net, config.TRAIN.OUTPUT_DIR)