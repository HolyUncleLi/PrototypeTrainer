import os
import json
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader
from train_mtcl import OneFoldTrainer
from models.protop_fusion import ProtoPNet

warnings.filterwarnings("ignore")


class OneFoldEvaluator(OneFoldTrainer):
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold

        self.cfg = config
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        self.criterion = nn.CrossEntropyLoss()
        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)

        # 超参
        self.λ_align = 1.0
        self.λ_band = 10.0
        self.λ_G_l1 = 1
        self.λ_F_l1 = 1
        self.λ_prior = 1
        self.λ_σ = 1
        self.f_min = 0.5
        self.f_max = 30.0

    def build_model(self):
        model = ProtoPNet(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model

    def build_dataloader(self):
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                                 num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True, drop_last=True)
        print('[INFO] Dataloader prepared')

        return {'test': test_loader}

    def run(self):
        print('\n[INFO] Fold: {}'.format(self.fold))
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        y_true, y_pred, mf1 = self.evaluate(mode='test')
        print('')

        return y_true, y_pred, mf1


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path',
                        default='./configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    config['mode'] = 'normal'

    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))
    cm = []

    for fold in range(1, config['dataset']['num_splits'] + 1):
        evaluator = OneFoldEvaluator(args, fold, config)
        y_true, y_pred, mf1 = evaluator.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])

        summarize_result(config, fold, Y_true, Y_pred)
        from visualize_prototype import visualize_filters_via_data
        
        visualize_filters_via_data(evaluator.model, evaluator.loader_dict['test'], evaluator.device)

        # cm.append(confusion_matrix(Y_true.astype(int), Y_pred.argmax(axis=1)))

    # 绘制平均混淆矩阵
    mean_cm = np.mean(cm, axis=0)
    # cm_plot(mean_cm, './results/cm.svg')


if __name__ == "__main__":
    main()

