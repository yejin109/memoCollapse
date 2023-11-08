import os
import ssl
import time
import pickle
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from logger import Logger
from argparse import Namespace
from sklearn.feature_selection import mutual_info_classif

import torch
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import get_model
from utils import Transform, LARS, exclude_bias_and_norm, adjust_learning_rate

ssl._create_default_https_context = ssl._create_unverified_context


parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', type=str, default='cuda:0')

# Logging
parser.add_argument('--logging_steps', type=int, default=5)
parser.add_argument('--saving_steps', type=int, default=500)
parser.add_argument('--output_dir', type=str, default='./output')


parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--backbone', type=str, default='resnet18')
parser.add_argument('--mode', type=str, default='simclr')

# Train
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)

# Distributed
parser.add_argument("--local-rank", type=int, default=0, help="Local rank. Necessary for using the torch.distributed.launch utility.")
parser.add_argument('--world_size', type=int, default=1)

# Metric
parser.add_argument('--smi_samples', type=int, default=30)


def compute_smi(_repre, _label, n_samples):
    def sampling():
        samples = np.random.randn(*_repre.shape)
        samples /= np.linalg.norm(samples, axis=0)
        return samples

    def samplewise_smi():
        samples = sampling()
        rv_x = (_repre * samples).sum(axis=1)
        rv_y = _label
        return mutual_info_classif(rv_x[:, np.newaxis], rv_y)

    smis = list(map(lambda x: samplewise_smi(), range(n_samples)))
    return np.mean(smis)


def compute_singular(latents):
    z = torch.nn.functional.normalize(latents, dim=1)

    # calculate covariance
    z = z.cpu().detach().numpy()
    z = np.transpose(z)
    c = np.cov(z)
    _, d, _ = np.linalg.svd(c)

    return np.log(d)


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    gpu = args.gpu
    logger = Logger(f'{str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))}', args.output_dir)
    logger.addFileHandler(f'train.txt')
    logger.addSteamHandler()

    if args.data == 'cifar10':
        dataset = CIFAR10(args.output_dir, transform=Transform(), download=True)
    else:
        raise KeyError()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, pin_memory=True)

    model = get_model(args.backbone, Namespace(**{'mode': args.mode, 'label_num': len(dataset.classes)}))
    model = model.to(args.gpu)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    global_steps = 0
    for epoch in range(args.epochs):
        for step, ((y1, y2), labels) in tqdm(enumerate(loader, start=epoch * len(loader)), desc='Iteration'):
            model.train()
            global_steps += 1
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu)
            lr = adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, acc = model.forward(y1, y2, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if global_steps % args.logging_steps == 0:
                model.eval()
                with torch.no_grad():
                    reprs = model.verbose_forward(y1)

                smi_log = {}
                for i, repre_layer in enumerate(reprs):
                    smi = compute_smi(repre_layer.flatten(start_dim=1).detach().cpu().numpy(), labels.detach().cpu().numpy(), args.smi_samples)
                    smi_log[f'Layer_{i}'] = smi

                emb_singular = compute_singular(reprs[-1].flatten(start_dim=1))

                # with open(f"{args.output_dir}/Epoch{epoch}-Gstep{global_steps}.pkl", "wb") as f:
                #     pickle.dump(reprs, f)
                stats = dict(epoch=epoch, step=step, learning_rate=lr,
                             loss=loss.item(), acc=acc.item(),
                             time=int(time.time() - start_time))
                stats = dict(stats, **smi_log)
                stats['Embedding_singular'] = np.sum(emb_singular>-8)
                log = '\n'
                for k, v in stats.items():
                    log += f'\t{k}:{v}\n'
                logger.debug(log)
