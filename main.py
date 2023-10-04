import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import argparse
import random
import numpy as np
from solver import Solver


import warnings
warnings.filterwarnings("ignore")

seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def main(config):
    img_num = {
        'live': list(range(1, 30)),
        'csiq': list(range(1, 31)),
    }
    sel_num = img_num[config.dataset]


    srcc_all = np.zeros(config.train_test_num, dtype=np.float)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    print("batch_size", config.batch_size)
    print("lr_1", config.lr_1)
    print("lr_2", config.lr_2)
    for i in range(config.train_test_num):
        print('Round %d' % (i + 1))
        random.shuffle(sel_num)
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        print("train_index", train_index)
        print("test_index", test_index)

        solver = Solver(config, train_index, test_index,  i)
        srcc_all[i], plcc_all[i] = solver.train()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='live')
    parser.add_argument('--lr_1', dest='lr_1', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lr_2', dest='lr_2', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='round of training and testing')
    parser.add_argument('--epochs', dest='epochs', type=int, default=30, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=112,
                        help='Crop size for training & testing image patches')

    config = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    main(config)
